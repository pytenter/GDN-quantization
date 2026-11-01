def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.access_token,self.expiration_unix_epoch,self.endpoint,)==(other.access_token,other.expiration_unix_epoch,other.endpoint,)
  return NotImplemented
 return __eq__