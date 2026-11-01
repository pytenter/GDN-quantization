def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.placeholders,self.stack_traces,self.cudagraph_fail_reasons,)==(other.placeholders,other.stack_traces,other.cudagraph_fail_reasons,)
  return NotImplemented
 return __eq__