def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.reason,self.user_stack,self.graph_break,)==(other.reason,other.user_stack,other.graph_break,)
  return NotImplemented
 return __eq__