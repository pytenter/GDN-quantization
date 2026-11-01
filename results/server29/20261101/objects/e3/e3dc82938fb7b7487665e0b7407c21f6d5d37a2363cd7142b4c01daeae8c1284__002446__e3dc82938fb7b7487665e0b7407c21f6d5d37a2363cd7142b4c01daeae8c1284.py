def __create_fn__():
 def __hash__(self):
  return hash((self.is_leaf,self.mutates_data,self.mutates_metadata,self.mutations_hidden_from_autograd,self.mutations_under_no_grad_or_inference_mode,self.mutation_inductor_storage_resize,self.mutates_storage_metadata,self.requires_grad,self.keep_input_mutations,))
 return __hash__