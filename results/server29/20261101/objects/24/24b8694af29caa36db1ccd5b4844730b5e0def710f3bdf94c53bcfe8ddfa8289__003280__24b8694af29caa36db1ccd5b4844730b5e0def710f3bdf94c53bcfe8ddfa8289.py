def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.loops,self.kernel,)==(other.loops,other.kernel,)
  return NotImplemented
 return __eq__