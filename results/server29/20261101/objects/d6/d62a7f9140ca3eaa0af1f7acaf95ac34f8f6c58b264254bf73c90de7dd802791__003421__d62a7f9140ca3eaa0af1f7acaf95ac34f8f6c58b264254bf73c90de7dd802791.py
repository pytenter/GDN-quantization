@torch.jit.script_method
def __getstate__(self):
    weight = self.weight.to_dense()
    bias = self.bias.to_dense()
    running_mean = self.running_mean.to_dense()
    running_var = self.running_var.to_dense()
    return (weight, bias, running_mean, running_var, self.training)
