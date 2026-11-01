def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.binary,self.properties,self.rendered,self.text,)==(other.binary,other.properties,other.rendered,other.text,)
  return NotImplemented
 return __eq__