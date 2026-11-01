def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.t_id,self.dim,)==(other.t_id,other.dim,)
  return NotImplemented
 return __eq__