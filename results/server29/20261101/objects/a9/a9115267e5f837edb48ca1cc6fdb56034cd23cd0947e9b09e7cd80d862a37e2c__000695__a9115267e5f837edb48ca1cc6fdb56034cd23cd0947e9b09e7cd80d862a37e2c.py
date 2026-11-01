def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.parent_fqn,self.parent_module,self.parent_call_module,self.fqn,self.call_idx,self.module,)==(other.parent_fqn,other.parent_module,other.parent_call_module,other.fqn,other.call_idx,other.module,)
  return NotImplemented
 return __eq__