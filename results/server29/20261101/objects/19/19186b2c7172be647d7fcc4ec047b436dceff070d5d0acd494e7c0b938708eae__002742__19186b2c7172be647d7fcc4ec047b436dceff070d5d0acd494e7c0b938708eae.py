def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.graph_module,self.opset_version,self.range_constraints,self.schema_version,self.verifiers,self.torch_version,)==(other.graph_module,other.opset_version,other.range_constraints,other.schema_version,other.verifiers,other.torch_version,)
  return NotImplemented
 return __eq__