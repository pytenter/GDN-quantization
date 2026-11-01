def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.kernel,self.structured,self.cpp_namespace,)==(other.kernel,other.structured,other.cpp_namespace,)
  return NotImplemented
 return __eq__