def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.name,self.constraint_range,self.val,)==(other.name,other.constraint_range,other.val,)
  return NotImplemented
 return __eq__