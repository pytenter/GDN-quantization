def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.message,self.nodeid,self.fslocation,)==(other.message,other.nodeid,other.fslocation,)
  return NotImplemented
 return __eq__