def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.graph,self.order,self.name,self.input_placeholders,self.orig_inputs,self.orig_outputs,self.getattr_maps,self.constructor_args,self.gm,)==(other.graph,other.order,other.name,other.input_placeholders,other.orig_inputs,other.orig_outputs,other.getattr_maps,other.constructor_args,other.gm,)
  return NotImplemented
 return __eq__