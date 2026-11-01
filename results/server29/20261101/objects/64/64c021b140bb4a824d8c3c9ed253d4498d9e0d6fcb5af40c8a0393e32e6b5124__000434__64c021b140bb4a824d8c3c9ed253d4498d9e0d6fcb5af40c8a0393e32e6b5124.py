def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.framework_loc,self.maybe_user_loc,)==(other.framework_loc,other.maybe_user_loc,)
  return NotImplemented
 return __eq__