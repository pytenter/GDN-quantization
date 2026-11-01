def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.text,self.markdown,self.properties,)==(other.text,other.markdown,other.properties,)
  return NotImplemented
 return __eq__