def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.token,self.compromise,self.heartbeat,)==(other.token,other.compromise,other.heartbeat,)
  return NotImplemented
 return __eq__