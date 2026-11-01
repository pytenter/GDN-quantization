def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.handle,self.actual_node_name,self.actual_module_stack,self.ref_node_name,self.ref_module_stack,self.results,)==(other.handle,other.actual_node_name,other.actual_module_stack,other.ref_node_name,other.ref_module_stack,other.results,)
  return NotImplemented
 return __eq__