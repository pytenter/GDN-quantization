def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.reads,self.writes,self.visited,)==(other.reads,other.writes,other.visited,)
  return NotImplemented
 return __eq__