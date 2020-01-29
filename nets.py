import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, bn=False):
        super().__init__()
        self.conv = nn.Conv2d(input_channel, output_channel, kernel_size, stride=1, padding=kernel_size//2, bias=False)
        self.bn = None
        if bn:
            self.bn = nn.BatchNorm2d(output_channel)

    def forward(self, x):
        h = self.conv(x)
        if self.bn is not None:
            h = self.bn(h)
        return h

class ResidualBlock(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, bn=False):
        super().__init__()
        self.conv = Conv(input_channel, output_channel, kernel_size, bn)

    def forward(self, x):
        return F.relu(x + (self.conv(x)))


num_filters = 8
num_blocks = 4
kernel_size = 3

class Representation(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.layer0 = Conv(input_shape[0], num_filters, 3, bn=True)
        self.blocks = nn.ModuleList([ResidualBlock(num_filters, num_filters, kernel_size=kernel_size) for _ in range(num_blocks)])
        self.linear = nn.Linear(input_shape[1]*input_shape[2]*num_filters, output_shape)

    def forward(self, x):
        h = F.relu(self.layer0(x))
        for block in self.blocks:
            h = block(h)
        shape = h.shape
        h = h.view(-1, shape[1]*shape[2]*shape[3])
        rp = self.linear(h)
        rp = F.relu(rp)
        return rp

    def inference(self, x):
        self.eval()
        x = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            rp = self(x)
        return rp.cpu().numpy()

class Prediction(nn.Module):
    def __init__(self, rp_shape, action_shape):
        super().__init__()
        self.latent_shape = 1024
        self.fc_l = nn.Linear(rp_shape, self.latent_shape)
        self.fc_p = nn.Linear(self.latent_shape, action_shape)
        self.fc_v = nn.Linear(self.latent_shape, 1)

    def forward(self, rp):
        rp_l = self.fc_l(rp)
        rp_l = F.relu(rp_l)
        h_p = self.fc_p(rp_l)
        h_v = self.fc_v(rp_l)
        return F.softmax(h_p), torch.tanh(h_v)

    def inference(self, rp):
        self.eval()
        rp = torch.tensor(rp, dtype=torch.float32)
        with torch.no_grad():
            p, v = self(rp)
        return p.cpu().numpy(), v.cpu().numpy()

class Dynamics(nn.Module):
    def __init__(self, rp_shape, action_shape):
        super().__init__()
        self.input_shape = rp_shape + action_shape
        self.latent_shape = 1024
        self.fc_l = nn.Linear(self.input_shape, self.latent_shape)
        self.fc_r = nn.Linear(self.latent_shape, 1)
        self.fc_h = nn.Linear(self.latent_shape, rp_shape)

    def forward(self, rp, a):
        input = torch.cat([rp, a], dim=1)
        l = self.fc_l(input)
        l = F.relu(l)
        h = self.fc_h(l)
        r = self.fc_r(l)
        return F.relu(h), torch.sigmoid(r)

    def inference(self, rp, a):
        self.eval()
        rp = torch.tensor(rp, dtype=torch.float32)
        a = torch.tensor(a, dtype=torch.float32)
        with torch.no_grad():
            rp, r = self(rp, a)
        return rp.cpu().numpy(), r.cpu().numpy()

class Nets(nn.Module):
    def __init__(self, state_shape, action_shape, rp_shape):
        super().__init__()
        self.representation = Representation(state_shape, rp_shape)
        self.prediction = Prediction(rp_shape, action_shape)
        self.dynamics = Dynamics(rp_shape, action_shape)
