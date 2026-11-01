def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.scope,self.params,self.autouse,self.ids,self.name,)==(other.scope,other.params,other.autouse,other.ids,other.name,)
  return NotImplemented
 return __eq__