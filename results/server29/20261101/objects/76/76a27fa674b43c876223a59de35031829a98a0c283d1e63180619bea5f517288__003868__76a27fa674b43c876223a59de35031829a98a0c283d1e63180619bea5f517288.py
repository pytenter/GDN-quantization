def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.id,self.url,self.job,self.watched,self.domains,self.secret,self.disabled,)==(other.id,other.url,other.job,other.watched,other.domains,other.secret,other.disabled,)
  return NotImplemented
 return __eq__