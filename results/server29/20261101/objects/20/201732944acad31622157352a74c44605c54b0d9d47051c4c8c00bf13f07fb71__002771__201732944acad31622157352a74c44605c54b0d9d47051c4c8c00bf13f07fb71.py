def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.graph_module,self.signature,self.module_call_graph,self.names_to_symbols,self.state_dict,self.constants,self.example_inputs,)==(other.graph_module,other.signature,other.module_call_graph,other.names_to_symbols,other.state_dict,other.constants,other.example_inputs,)
  return NotImplemented
 return __eq__