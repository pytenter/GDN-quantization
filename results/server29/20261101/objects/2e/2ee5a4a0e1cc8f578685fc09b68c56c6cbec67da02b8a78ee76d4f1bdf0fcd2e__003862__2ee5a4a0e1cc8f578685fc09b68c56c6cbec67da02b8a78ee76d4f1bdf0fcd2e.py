def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.username,self.fullname,self.email,self.timestamp,self.status,self.fields,)==(other.username,other.fullname,other.email,other.timestamp,other.status,other.fields,)
  return NotImplemented
 return __eq__