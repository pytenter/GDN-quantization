def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.oid,self.title,self.date,)==(other.oid,other.title,other.date,)
  return NotImplemented
 return __eq__