def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.guard_export_fn,self.guard_fail_fn,)==(other.guard_export_fn,other.guard_fail_fn,)
  return NotImplemented
 return __eq__