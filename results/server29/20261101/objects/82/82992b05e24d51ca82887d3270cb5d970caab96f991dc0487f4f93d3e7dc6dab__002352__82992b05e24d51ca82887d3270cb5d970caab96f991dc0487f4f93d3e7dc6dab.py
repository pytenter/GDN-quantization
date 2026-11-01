def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.lock_file,self.token,self.reason,self.error,)==(other.lock_file,other.token,other.reason,other.error,)
  return NotImplemented
 return __eq__