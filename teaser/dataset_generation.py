import scipy
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import math
import numpy as np
import argparse
from tqdm import tqdm

def simple_pendulum_deriv(x, t, m, g, l, F, c, omega): 
# The simple pendulum subject to zero damping and zero control input 
    nx = np.zeros(2)
    nx[0] = x[1]
    nx[1] = (1/m) * (F * math.sin(omega * t) - (m * g / l) * x[0] - c * nx[0])
    return nx

def generate_dissipative_sets_for_pendulum(c_array):
    t_span = np.linspace(0,50,750)
    sols = []
    positions = np.linspace(-math.pi,math.pi,100)
    np.random.shuffle(positions)
    vels = np.linspace(-2,2,100)
    np.random.shuffle(vels)
    for start_pos in tqdm(positions):
        part_c = []
        for start_vel in vels:
            sol = odeint(simple_pendulum_deriv, y0=[start_pos,start_vel], t=t_span, args=(1,9.8,1,0.6,0,1))
            part_c.append(sol)
        
        sols.append(part_c)
    
    sols_array = np.array(sols)
    # noise_array = sols_array
    np.save(f'pendulum-{args.c_domain}-{args.c_points}', sols_array)
    
    return sols_array

if __name__ == '__main__':
        # TRAINING ARGUMENTS
    parser = argparse.ArgumentParser(description='Autoencoder Prediction')
    #
    parser.add_argument('--c_domain', type=float, default='0.6', help='domain of c values')
    #
    parser.add_argument('--c_points', type=int, default='10', help='number of points in c domain')

    args = parser.parse_args()
    c_array = np.linspace(-args.c_domain, args.c_domain, args.c_points)
    print(c_array)
    examples = generate_dissipative_sets_for_pendulum(c_array)
    print(examples.shape)