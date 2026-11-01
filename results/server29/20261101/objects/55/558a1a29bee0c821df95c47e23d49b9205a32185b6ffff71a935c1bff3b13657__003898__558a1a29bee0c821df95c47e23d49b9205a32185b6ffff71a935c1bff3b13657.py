def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.name,self.ref,self.target_commit,)==(other.name,other.ref,other.target_commit,)
  return NotImplemented
 return __eq__