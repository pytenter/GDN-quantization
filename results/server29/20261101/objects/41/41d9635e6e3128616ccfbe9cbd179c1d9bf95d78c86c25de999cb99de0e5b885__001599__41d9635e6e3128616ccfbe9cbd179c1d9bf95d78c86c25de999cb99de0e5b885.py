def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.code_list,self.guard,)==(other.code_list,other.guard,)
  return NotImplemented
 return __eq__