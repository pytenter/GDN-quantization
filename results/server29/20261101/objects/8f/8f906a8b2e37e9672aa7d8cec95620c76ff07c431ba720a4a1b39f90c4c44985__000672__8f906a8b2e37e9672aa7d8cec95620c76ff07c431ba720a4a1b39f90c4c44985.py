def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.inputs,self.outputs,self.in_spec,self.out_spec,self.forward_arg_names,)==(other.inputs,other.outputs,other.in_spec,other.out_spec,other.forward_arg_names,)
  return NotImplemented
 return __eq__