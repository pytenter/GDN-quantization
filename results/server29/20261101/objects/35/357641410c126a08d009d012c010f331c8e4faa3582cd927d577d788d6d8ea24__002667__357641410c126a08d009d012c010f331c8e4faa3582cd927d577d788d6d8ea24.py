def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.target,self.inputs,self.outputs,self.metadata,self.is_hop_single_tensor_return,)==(other.target,other.inputs,other.outputs,other.metadata,other.is_hop_single_tensor_return,)
  return NotImplemented
 return __eq__