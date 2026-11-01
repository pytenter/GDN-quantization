def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.reads,self.writes,self.memory,)==(other.reads,other.writes,other.memory,)
  return NotImplemented
 return __eq__