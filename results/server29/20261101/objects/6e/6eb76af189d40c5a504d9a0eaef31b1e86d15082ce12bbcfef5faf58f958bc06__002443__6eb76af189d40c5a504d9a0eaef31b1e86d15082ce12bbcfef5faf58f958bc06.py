def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.is_leaf,self.mutates_data,self.mutates_metadata,self.mutations_hidden_from_autograd,self.mutations_under_no_grad_or_inference_mode,self.mutation_inductor_storage_resize,self.mutates_storage_metadata,self.requires_grad,self.keep_input_mutations,)==(other.is_leaf,other.mutates_data,other.mutates_metadata,other.mutations_hidden_from_autograd,other.mutations_under_no_grad_or_inference_mode,other.mutation_inductor_storage_resize,other.mutates_storage_metadata,other.requires_grad,other.keep_input_mutations,)
  return NotImplemented
 return __eq__