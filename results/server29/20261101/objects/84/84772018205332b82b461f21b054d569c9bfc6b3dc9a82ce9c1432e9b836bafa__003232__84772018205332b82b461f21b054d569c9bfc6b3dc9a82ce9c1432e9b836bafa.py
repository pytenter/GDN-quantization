def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.wrapper,self.node,self.reused_as,self.delete_old,)==(other.wrapper,other.node,other.reused_as,other.delete_old,)
  return NotImplemented
 return __eq__