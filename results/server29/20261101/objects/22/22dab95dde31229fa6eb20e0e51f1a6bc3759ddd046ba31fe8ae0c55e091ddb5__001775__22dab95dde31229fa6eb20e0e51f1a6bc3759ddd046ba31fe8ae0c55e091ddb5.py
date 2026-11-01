def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.reduce_op,self.norm_type,)==(other.reduce_op,other.norm_type,)
  return NotImplemented
 return __eq__