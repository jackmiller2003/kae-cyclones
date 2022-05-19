from torch import nn
import torch
import torch.nn.functional as F

def gaussian_init_(n_units, std=1):    
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std/n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]  
    return Omega

class encoderNet(nn.Module):
    def __init__(self, alpha, b):
        super(encoderNet, self).__init__()

        self.tanh = nn.Tanh()

        self.conv1 = nn.Conv2d(in_channels=20, out_channels=256, kernel_size=11, stride=4, padding=0, groups=1, bias=True)
        self.conv1_bn = nn.BatchNorm2d(256)

        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=0, groups=1, bias=True)
        self.conv2_bn = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2, stride=1, padding=0, groups=1, bias=True)
        self.conv3_bn = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(256*2*2, 16 * alpha)
        self.fc2 = nn.Linear(16 * alpha, 16 * alpha)
        self.fc2 = nn.Linear(16 * alpha, 16 * alpha)
        self.fc3 = nn.Linear(16 * alpha, b)

        self.init_weights()

    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):

        x = self.conv1_bn(self.conv1(x))
        # print(f"4. before last conv {x.shape}")
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=0)

        # print(f"3. before last conv {x.shape}")
        x = self.conv2_bn(self.conv2(x))
        # print(f"2. before last conv {x.shape}")
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=0)

        # print(f"before last conv {x.shape}")
        x = self.conv3_bn(self.conv3(x))

        x = x.view(-1, 256*2*2)

        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)

        return x


class decoderNet(nn.Module):
    def __init__(self, alpha, b):
        super(decoderNet, self).__init__()

        self.tanh = nn.Tanh()

        self.fc1 = nn.Linear(b, 16*alpha)
        self.fc2 = nn.Linear(16*alpha, 16*alpha)
        self.fc3 = nn.Linear(16*alpha, 256*2*2)

        self.convtrans1 = nn.ConvTranspose2d(in_channels = 256, out_channels = 256, kernel_size=2, stride=1, padding=0, bias=True)
        self.convtrans1_bn = nn.BatchNorm2d(256)
        self.convtrans2 = nn.ConvTranspose2d(in_channels = 256, out_channels = 256, kernel_size=3, stride=2, padding=0, output_padding=1, bias=True)
        self.convtrans2_bn = nn.BatchNorm2d(256)
        self.convtrans3 = nn.ConvTranspose2d(in_channels = 256, out_channels = 256, kernel_size=3, stride=2, padding=0, output_padding=1, bias=True)
        self.convtrans3_bn = nn.BatchNorm2d(256)
        self.convtrans4 = nn.ConvTranspose2d(in_channels = 256, out_channels = 256, kernel_size=3, stride=2, padding=0, output_padding=1, bias=True)
        self.convtrans4_bn = nn.BatchNorm2d(256)
        self.convtrans5 = nn.ConvTranspose2d(in_channels = 256, out_channels = 20, kernel_size=11, stride=4, padding=0, output_padding=1, bias=True)
        self.convtrans5_bn = nn.BatchNorm2d(20)
        
        self.init_weights()

    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):

        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)

        x = torch.reshape(x,(1,256,2,2))
        
        x = self.convtrans1_bn(self.convtrans1(x))
        # print(f"After first inverse {x.shape}")
        x = self.convtrans2_bn(self.convtrans2(x))
        # print(f"After second inverse {x.shape}")
        x = self.convtrans3_bn(self.convtrans3(x))
        # print(f"After third inverse {x.shape}")
        x = self.convtrans4_bn(self.convtrans4(x))
        # print(f"After fourth inverse {x.shape}")
        x = self.convtrans5_bn(self.convtrans5(x))

        return x

class dynamics(nn.Module):
    def __init__(self, b, init_scale):
        super(dynamics, self).__init__()
        self.dynamics = nn.Linear(b, b, bias=False)
        self.dynamics.weight.data = gaussian_init_(b, std=1)           
        U, _, V = torch.svd(self.dynamics.weight.data)
        self.dynamics.weight.data = torch.mm(U, V.t()) * init_scale
    
    def forward(self, x):
        x = self.dynamics(x)
        return x

class dynamics_back(nn.Module):
    def __init__(self, b, omega):
        super(dynamics_back, self).__init__()
        self.dynamics = nn.Linear(b, b, bias=False)
        self.dynamics.weight.data = torch.pinverse(omega.dynamics.weight.data.t())     

    def forward(self, x):
        x = self.dynamics(x)
        return x

class koopmanAE(nn.Module):
    def __init__(self, b, steps, steps_back, alpha = 1, init_scale=1):
        super(koopmanAE, self).__init__()
        self.steps = steps
        self.steps_back = steps_back
        
        self.encoder = encoderNet(alpha = alpha, b=b)
        self.dynamics = dynamics(b, init_scale)
        self.backdynamics = dynamics_back(b, self.dynamics)
        self.decoder = decoderNet(alpha = alpha, b=b)


    def forward(self, x, mode='forward'):
        out = []
        out_back = []
        z = self.encoder(x.contiguous())
        q = z.contiguous()

        
        if mode == 'forward':
            for _ in range(self.steps):
                q = self.dynamics(q)
                out.append(self.decoder(q))

            out.append(self.decoder(z.contiguous())) 
            return out, out_back    

        if mode == 'backward':
            for _ in range(self.steps_back):
                q = self.backdynamics(q)
                out_back.append(self.decoder(q))
                
            out_back.append(self.decoder(z.contiguous()))
            return out, out_back