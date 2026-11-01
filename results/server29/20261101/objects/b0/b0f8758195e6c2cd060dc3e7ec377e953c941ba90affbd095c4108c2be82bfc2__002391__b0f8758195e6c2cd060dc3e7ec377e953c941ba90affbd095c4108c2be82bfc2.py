def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.name,self.state,self.token,self.pid,self.hostname,self.start,)==(other.name,other.state,other.token,other.pid,other.hostname,other.start,)
  return NotImplemented
 return __eq__