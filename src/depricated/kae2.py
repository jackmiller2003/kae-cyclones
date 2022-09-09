class koopmanAE2(nn.Module):
    def __init__(self, b, steps, steps_back, alpha = 4, init_scale=10, simple=True, norm=True, print_hidden=False, maxmin=2, eigen_init=True, eigen_distribution='uniform', input_size=400, std=1):
        super(koopmanAE, self).__init__()
        self.steps = steps
        self.steps_back = steps_back

        if simple:
            self.encoder = encoderNetSimple(alpha = alpha, b=b, input_size=input_size)
            self.decoder = decoderNetSimple(alpha = alpha, b=b, input_size=input_size)
        else:
            self.encoder = encoderNet(alpha = alpha, b=b)
            self.decoder = decoderNet(alpha = alpha, b=b)
        
        self.dynamics = dynamics(b, init_scale, eigen_init=eigen_init, maxmin=maxmin, eigen_distribution=eigen_distribution, std=std)
        self.backdynamics = dynamics_back(b, self.dynamics)
        self.print_hidden = print_hidden


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