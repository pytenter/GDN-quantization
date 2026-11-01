def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.nodes,self.source,self.input_nodes,self.output_nodes,self.params,)==(other.nodes,other.source,other.input_nodes,other.output_nodes,other.params,)
  return NotImplemented
 return __eq__