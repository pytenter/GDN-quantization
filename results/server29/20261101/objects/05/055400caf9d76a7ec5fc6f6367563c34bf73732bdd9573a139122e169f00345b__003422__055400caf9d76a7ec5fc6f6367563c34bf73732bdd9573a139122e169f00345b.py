@torch.jit.script_method
def __setstate__(self, state):
    self.weight = state[0].to_mkldnn()
    self.bias = state[1].to_mkldnn()
    self.running_mean = state[2].to_mkldnn()
    self.running_var = state[3].to_mkldnn()
    self.training = state[4]
