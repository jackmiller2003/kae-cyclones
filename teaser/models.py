from torch import nn
import torch
import torch.nn.functional as F
from initLibrary import *
from experiment import *
# from eunn import EUNN
import torch.nn.utils.parametrizations as tparam

def gaussian_init_(n_units, std=1):    
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std/n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]
    w, v = np.linalg.eig(Omega.cpu().detach().numpy())
    return Omega

def eigen_init_(n_units, distribution='uniform',std=1, maxmin=2):
    # Orthogoonal matrices
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std/n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]  
    w, v = np.linalg.eig(Omega.cpu().detach().numpy())

    if distribution == 'uniform':
        print(f"Uniform {maxmin}")
        w.real = np.random.uniform(-maxmin,maxmin, w.shape[0])
        imag_dist = np.random.uniform(-maxmin,maxmin, w.shape[0])
        w = w + np.zeros(w.shape[0], dtype=complex)
        w.imag = imag_dist
        print(w)
    elif distribution == 'uniform-small':
        w.real = np.random.uniform(-1,1, w.shape[0])
        imag_dist = np.random.uniform(-1,1, w.shape[0])
        w = w + imag_dist
    elif distribution == 'gaussian':
        w.real = np.random.normal(loc=0, scale=std, size=w.shape[0])
        imag_dist = np.random.normal(loc=0, scale=std, size=w.shape[0])
        w = w + imag_dist
    elif distribution == 'double-gaussian':
        w.real = np.random.normal(loc=1, scale=std, size=w.shape[0]) + np.random.normal(loc=-1, scale=std, size=w.shape[0])
        imag_dist = np.random.normal(loc=1, scale=std, size=w.shape[0]) + np.random.normal(loc=-1, scale=std, size=w.shape[0])
        w = w + imag_dist
    
    return torch.from_numpy(reconstruct_operator(w,v).real).float()

class encoderNetSimple(nn.Module):
    def __init__(self, alpha, b, input_size=2, spectral_norm=False):
        super(encoderNetSimple, self).__init__()
        self.input_size = input_size

        if spectral_norm:
            self.fc1 = tparam.spectral_norm(nn.Linear(self.input_size, 16 * alpha))
            self.fc2 = tparam.spectral_norm(nn.Linear(16 * alpha, 16 * alpha))
            self.fc3 = tparam.spectral_norm(nn.Linear(16 * alpha, b))
        else:
            self.fc1 = nn.Linear(self.input_size, 16 * alpha)
            self.fc2 = nn.Linear(16 * alpha, 16 * alpha)
            self.fc3 = nn.Linear(16 * alpha, b)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)

        return x

class decoderNetSimple(nn.Module):
    def __init__(self, alpha, b, spectral_norm=False, input_size=2):
        super(decoderNetSimple, self).__init__()
        self.b = b

        self.input_size = input_size

        if spectral_norm:
            self.fc1 = tparam.spectral_norm(nn.Linear(b, 16 * alpha))
            self.fc2 = tparam.spectral_norm(nn.Linear(16 * alpha, 16 * alpha))
            self.fc3 = tparam.spectral_norm(nn.Linear(16 * alpha, self.input_size))
        else:
            self.fc1 = nn.Linear(b, 16 * alpha)
            self.fc2 = nn.Linear(16 * alpha, 16 * alpha)
            self.fc3 = nn.Linear(16 * alpha, self.input_size)      

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = x.view(-1, 1, self.b)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)

        if self.input_size == 400:
            return x.view(-1, 1, 20, 20)
        else:
            return x.view(-1, 1, self.input_size)

class dynamics(nn.Module):
    def __init__(self, b, init_scheme, spectral_norm=False):
        super(dynamics, self).__init__()
        self.dynamics = nn.Linear(b, b, bias=False)
        self.dynamics.weight.data = init_scheme()

    def forward(self, x):
        x = self.dynamics(x)
        return x

class dynamics_back(nn.Module):
    def __init__(self, b, init_scheme):
        super(dynamics_back, self).__init__()
        self.dynamics = nn.Linear(b, b, bias=False)
        self.dynamics.weight.data = init_scheme()

    def forward(self, x):
        x = self.dynamics(x)
        return x

class koopmanAE(nn.Module):
    def __init__(self, init_scheme, b, alpha = 4, input_size=400, spectral_norm=False, steps=4, back=False):
        super(koopmanAE, self).__init__()
        self.steps = steps
        self.approx_steps = 4
        self.steps_back = steps
        self.encoder = encoderNetSimple(alpha = alpha, b=b, input_size=input_size)
        self.decoder = decoderNetSimple(alpha = alpha, b=b, input_size=input_size)
        self.back = back
        self.dynamics = dynamics(b, init_scheme, False)
        self.backdynamics = dynamics_back(b, init_scheme)


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
                #print(q)
                out_back.append(self.decoder(q))
            
            out_back.append(self.decoder(z.contiguous()))
            return out, out_back
        
        if mode =='forward-approx':
            stepList = np.linspace(0,self.steps, 4, dtype=int)
            q = torch.squeeze(q)
            for step in stepList:
                powermatrix = torch.linalg.matrix_power(self.dynamics.dynamics.weight, step)
                out.append(self.decoder(torch.linalg.multi_dot([powermatrix, q])))
            
            out.append(self.decoder(z.contiguous())) 
            return out, out_back
        
        if mode =='backward-approx':
            q = torch.squeeze(q)
            stepList = np.linspace(0,self.steps, 4, dtype=int)
            for step in stepList:
                powermatrix = torch.linalg.matrix_power(self.backdynamics.dynamics.weight, step)
                out_back.append(self.decoder(torch.linalg.multi_dot([powermatrix, q])))
            
            out_back.append(self.decoder(z.contiguous()))
            return out, out_back

class koopmanAE2(nn.Module):
    def __init__(self, b, steps, steps_back, alpha = 4, init_scale=10, simple=True, norm=True, print_hidden=False, maxmin=2, eigen_init=True, eigen_distribution='uniform', input_size=400, std=1):
        super(koopmanAE, self).__init__()
        self.steps = steps
        self.steps_back = steps_back

        self.dynamics = dynamics(b, init_scheme, spectral_norm=spectral_norm)
        self.backdynamics = dynamics_back(b, self.dynamics)


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

class nonlinearDynamics(nn.Module):
    def __init__(self, b, init_scheme, spectral_norm=False):
        super(nonlinearDynamics, self).__init__()
        self.dynamics = nn.Linear(b, b)
        print(self.dynamics.weight)

    def forward(self, x):
        x = F.leaky_relu(self.dynamics(x))
        return x

class nonlinearDynamicsBack(nn.Module):
    def __init__(self, b, omega):
        super(nonlinearDynamicsBack, self).__init__()
        self.dynamics = nn.Linear(b, b)
        print(self.dynamics.weight)
        #self.dynamics.weight.data = torch.pinverse(omega.dynamics.weight.data.t())

    def forward(self, x):
        x = F.leaky_relu(self.dynamics(x))
        return x
        
class regularAE(nn.Module):
    def __init__(self, init_scheme, b, alpha = 4, input_size=400, spectral_norm=False, steps=4):
        super(regularAE, self).__init__()
        self.steps = steps
        self.steps_back = 4
        self.encoder = encoderNetSimple(alpha = alpha, b=b, input_size=input_size)
        self.decoder = decoderNetSimple(alpha = alpha, b=b, input_size=input_size)
        self.dynamics = nonlinearDynamics(b, init_scheme, False)
        self.backdynamics = nonlinearDynamicsBack(b, self.dynamics)


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

class feedForward(nn.Module):
    def __init__(self, init_scheme, b, alpha = 4, input_size=400, spectral_norm=False, steps=4):
        super(feedForward, self).__init__()
        self.steps = steps
        self.steps_back = 4
        self.encoder = encoderNetSimple(alpha = alpha, b=b, input_size=input_size)
        self.decoder = decoderNetSimple(alpha = alpha, b=b, input_size=input_size)
        self.dynamics = nonlinearDynamics(alpha*16, init_scheme, False)
        self.backdynamics = nonlinearDynamicsBack(b, self.dynamics)


    def forward(self, x, mode='forward'):
        out = []
        out_back = []
        z = self.encoder(x.contiguous())
        q = z.contiguous()

        
        if mode == 'forward':
            for _ in range(self.steps):
                q = self.decoder(self.encoder(q))
                out.append(q)

            return out, out_back    

        if mode == 'backward':
            for _ in range(self.steps_back):
                q = self.backdynamics(q)
                out_back.append(self.decoder(q))
                
            out_back.append(self.decoder(z.contiguous()))
            return out, out_back

class predictionANN(nn.Module):
    def __init__(self, alpha):
        super(predictionANN, self).__init__()

        self.fc1 = nn.Linear(400, 16 * alpha)
        self.fc2 = nn.Linear(16 * alpha, 16 * alpha)
        self.fc3 = nn.Linear(16 * alpha, 2)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = x.view(-1, 400)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)

        return x