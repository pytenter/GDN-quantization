def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.view,self.view_copy,self.view_inplace,)==(other.view,other.view_copy,other.view_inplace,)
  return NotImplemented
 return __eq__