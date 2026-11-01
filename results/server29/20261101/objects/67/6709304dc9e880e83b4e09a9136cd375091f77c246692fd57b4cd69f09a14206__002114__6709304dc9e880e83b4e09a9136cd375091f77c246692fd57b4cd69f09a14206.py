def __create_fn__():
 def __hash__(self):
  return hash((self.onnx_function,self.op_full_name,self.is_custom,self.is_complex,))
 return __hash__