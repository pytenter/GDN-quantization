def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.t_id,self.dim,self.name,self.constraint_range,self.root,self.fn,)==(other.t_id,other.dim,other.name,other.constraint_range,other.root,other.fn,)
  return NotImplemented
 return __eq__