def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.static_argnum,self.static_kwargkey,self.needs_pytree,)==(other.static_argnum,other.static_kwargkey,other.needs_pytree,)
  return NotImplemented
 return __eq__