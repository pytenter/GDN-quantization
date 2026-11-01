def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.graph,self.signature,self.module_call_graph,self.metadata,self.treespec_namedtuple_fields,)==(other.graph,other.signature,other.module_call_graph,other.metadata,other.treespec_namedtuple_fields,)
  return NotImplemented
 return __eq__