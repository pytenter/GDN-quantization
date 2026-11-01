def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.start,self.end,self.target,self.depth,self.lasti,)==(other.start,other.end,other.target,other.depth,other.lasti,)
  return NotImplemented
 return __eq__