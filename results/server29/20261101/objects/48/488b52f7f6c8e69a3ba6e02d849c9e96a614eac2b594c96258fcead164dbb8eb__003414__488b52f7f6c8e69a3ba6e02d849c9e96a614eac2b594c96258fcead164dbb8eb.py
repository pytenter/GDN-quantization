@torch.jit.script_method
def __setstate__(self, state):
    self.weight = state[0].to_mkldnn()
    self.bias = state[1].to_mkldnn()
    self.training = state[2]
