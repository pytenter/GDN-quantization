def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.arguments,self.id,self.markdown,self.properties,self.text,)==(other.arguments,other.id,other.markdown,other.properties,other.text,)
  return NotImplemented
 return __eq__