def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.code,self.guard_manager,self.compile_id,self.trace_annotation,)==(other.code,other.guard_manager,other.compile_id,other.trace_annotation,)
  return NotImplemented
 return __eq__