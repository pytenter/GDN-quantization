def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.bw_module,self.placeholder_list,self.saved_context,self.saved_compile_context,)==(other.bw_module,other.placeholder_list,other.saved_context,other.saved_compile_context,)
  return NotImplemented
 return __eq__