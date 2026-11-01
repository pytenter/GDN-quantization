def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.pattern,self.extra_check,self.normalize_args,)==(other.pattern,other.extra_check,other.normalize_args,)
  return NotImplemented
 return __eq__