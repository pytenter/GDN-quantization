def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.return_new_outs,)==(other.return_new_outs,)
  return NotImplemented
 return __eq__