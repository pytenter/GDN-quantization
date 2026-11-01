@torch.jit.script_method
def __setstate__(self, state):
    self.weight = torch._C._nn.mkldnn_reorder_conv2d_weight(
        state[0].to_mkldnn(),
        self.padding,
        self.stride,
        self.dilation,
        self.groups)
    self.bias = state[1].to_mkldnn()
    self.training = state[2]
