def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.location,self.module,self.parameters,self.properties,self.thread_id,)==(other.location,other.module,other.parameters,other.properties,other.thread_id,)
  return NotImplemented
 return __eq__