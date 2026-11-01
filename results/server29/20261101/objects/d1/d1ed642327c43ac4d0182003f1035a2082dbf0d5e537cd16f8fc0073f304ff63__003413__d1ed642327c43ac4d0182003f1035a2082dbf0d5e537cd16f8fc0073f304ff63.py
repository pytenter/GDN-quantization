@torch.jit.script_method
def __getstate__(self):
    return (self.weight.to_dense(), self.bias.to_dense(), self.training)
