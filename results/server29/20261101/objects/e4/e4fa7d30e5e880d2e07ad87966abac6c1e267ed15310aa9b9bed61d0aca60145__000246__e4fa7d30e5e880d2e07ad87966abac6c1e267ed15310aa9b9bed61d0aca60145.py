def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.type,self.context,self.children_spec,)==(other.type,other.context,other.children_spec,)
  return NotImplemented
 return __eq__