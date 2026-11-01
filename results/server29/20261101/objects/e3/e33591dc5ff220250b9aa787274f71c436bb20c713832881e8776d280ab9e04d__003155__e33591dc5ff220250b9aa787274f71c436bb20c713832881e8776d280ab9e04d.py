def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.processes,self.executor,)==(other.processes,other.executor,)
  return NotImplemented
 return __eq__