def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.inplace_op,self.mutated_arg,self.extra_check,)==(other.inplace_op,other.mutated_arg,other.extra_check,)
  return NotImplemented
 return __eq__