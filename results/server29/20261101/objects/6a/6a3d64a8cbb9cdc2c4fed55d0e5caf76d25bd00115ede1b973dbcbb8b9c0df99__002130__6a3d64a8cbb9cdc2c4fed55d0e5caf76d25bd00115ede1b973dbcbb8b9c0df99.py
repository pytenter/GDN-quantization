def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.fake,self.source,self.symbolic_context,)==(other.fake,other.source,other.symbolic_context,)
  return NotImplemented
 return __eq__