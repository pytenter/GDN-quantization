def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.locations,self.id,self.immutable_state,self.initial_state,self.message,self.properties,)==(other.locations,other.id,other.immutable_state,other.initial_state,other.message,other.properties,)
  return NotImplemented
 return __eq__