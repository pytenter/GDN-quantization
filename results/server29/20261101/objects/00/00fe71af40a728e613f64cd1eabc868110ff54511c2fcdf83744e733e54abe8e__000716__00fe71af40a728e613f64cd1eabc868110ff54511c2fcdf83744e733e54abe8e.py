def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.keyset,self.keyword_only_args,)==(other.keyset,other.keyword_only_args,)
  return NotImplemented
 return __eq__