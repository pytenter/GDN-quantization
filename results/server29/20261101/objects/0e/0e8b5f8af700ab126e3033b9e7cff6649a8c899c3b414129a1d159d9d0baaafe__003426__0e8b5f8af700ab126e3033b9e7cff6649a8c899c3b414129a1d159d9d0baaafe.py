@torch.jit.script_method
def forward(self, x):
    x_mkldnn = x if x.is_mkldnn else x.to_mkldnn()
    y_mkldnn = torch.prelu(x_mkldnn, self.weight)
    y = y_mkldnn if x.is_mkldnn else y_mkldnn.to_dense()
    return y
