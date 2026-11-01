def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.input_nodes,self.output_nodes,self.input_deallocation,self.skip_cudagraph,)==(other.input_nodes,other.output_nodes,other.input_deallocation,other.skip_cudagraph,)
  return NotImplemented
 return __eq__