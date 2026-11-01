@torch.jit.script_method
def forward(self, x):
    return torch.mkldnn_convolution(
        x,
        self.weight,
        self.bias,
        self.padding,
        self.stride,
        self.dilation,
        self.groups)
