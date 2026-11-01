def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.verbosity_level,self.warnings_as_errors,)==(other.verbosity_level,other.warnings_as_errors,)
  return NotImplemented
 return __eq__