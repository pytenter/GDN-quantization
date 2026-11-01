def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.onnx_function,self.op_full_name,self.is_custom,self.is_complex,)==(other.onnx_function,other.op_full_name,other.is_custom,other.is_complex,)
  return NotImplemented
 return __eq__