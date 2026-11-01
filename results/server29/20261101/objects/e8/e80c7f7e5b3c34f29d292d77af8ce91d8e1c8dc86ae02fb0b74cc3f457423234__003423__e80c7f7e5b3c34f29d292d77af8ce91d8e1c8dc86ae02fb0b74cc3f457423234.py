@torch.jit.script_method
def forward(self, x):
    return torch.batch_norm(
        x,
        self.weight,
        self.bias,
        self.running_mean,
        self.running_var,
        False,  # training
        self.exponential_average_factor,
        self.eps,
        False,  # cuda_enabled
    )
