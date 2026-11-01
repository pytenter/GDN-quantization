def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.bottom,self.left,self.message,self.properties,self.right,self.top,)==(other.bottom,other.left,other.message,other.properties,other.right,other.top,)
  return NotImplemented
 return __eq__