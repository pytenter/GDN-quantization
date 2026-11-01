def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.backend,self.init_method,self.timeout,)==(other.backend,other.init_method,other.timeout,)
  return NotImplemented
 return __eq__