import torch
import math
import numpy as np

class L2_Dist_Func_Mae(torch.nn.Module):
    def __init__(self):
        super(L2_Dist_Func_Mae, self).__init__()
    
    def forward(self, output_tensor:torch.Tensor, target_tensor: torch.Tensor):
        """
        Here we have two inputs:
            * Predicted location -> (lon disp., lat disp., change in intensity)
            * Target location -> (lon (t-1, t), lat (t-1, t), intensity (t-1,t))
        """

        pred_location = target_tensor[:,0:2,0] + output_tensor[:,0:2]
        true_location = target_tensor[:,0:2,1]

        # R = 6371e3
        R = 6371 # in km

        lon0, lat0 = true_location[:,0], true_location[:,1]
        lon1,lat1 = pred_location[:,0], pred_location[:,1]

        phi0 = lat0 * (math.pi/180) # Rads
        phi1 = lat1 * (math.pi/180)

        delta_phi = phi1 - phi0
        delta_lambda = (lon1 - lon0) * (math.pi/180)

        a = torch.pow(torch.sin(delta_phi/2),2) + torch.cos(phi0) * torch.cos(phi1) * torch.pow(torch.sin(delta_lambda/2),2) 
        a = a.float()
        c = 2 * R * torch.atan2(torch.sqrt(a), torch.sqrt(1-a))
        c = c.double()
        
        mean_dist_loss = torch.sum(c)/output_tensor.shape[0]
        
        loss_out = mean_dist_loss

        return loss_out