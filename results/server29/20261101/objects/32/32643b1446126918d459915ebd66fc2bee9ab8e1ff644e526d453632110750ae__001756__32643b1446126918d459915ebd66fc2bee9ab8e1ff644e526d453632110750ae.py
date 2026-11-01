def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.compute_mesh,self.schema,self.flat_args_schema,self.local_args,self.local_kwargs,self.args_tree_spec,self.output_sharding,)==(other.compute_mesh,other.schema,other.flat_args_schema,other.local_args,other.local_kwargs,other.args_tree_spec,other.output_sharding,)
  return NotImplemented
 return __eq__