def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.scheduling,self.wrapper_codegen,self.cpp_wrapper_codegen,)==(other.scheduling,other.wrapper_codegen,other.cpp_wrapper_codegen,)
  return NotImplemented
 return __eq__