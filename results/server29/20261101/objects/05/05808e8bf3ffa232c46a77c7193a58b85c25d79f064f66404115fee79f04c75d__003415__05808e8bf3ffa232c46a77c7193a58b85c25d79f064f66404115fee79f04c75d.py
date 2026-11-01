@torch.jit.script_method
def forward(self, x):
    x_mkldnn = x if x.is_mkldnn else x.to_mkldnn()
    y_mkldnn = torch._C._nn.mkldnn_linear(x_mkldnn, self.weight, self.bias)
    y = y_mkldnn if x.is_mkldnn else y_mkldnn.to_dense()
    return y
