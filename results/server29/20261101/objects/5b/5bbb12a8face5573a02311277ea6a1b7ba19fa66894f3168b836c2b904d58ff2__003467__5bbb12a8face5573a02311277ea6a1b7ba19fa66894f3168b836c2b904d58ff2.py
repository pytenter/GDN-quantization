def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.category,self.template,)==(other.category,other.template,)
  return NotImplemented
 return __eq__