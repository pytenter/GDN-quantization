def __create_fn__():
 def __repr__(self):
  return self.__class__.__qualname__ + f"(is_leaf={self.is_leaf!r}, mutates_data={self.mutates_data!r}, mutates_metadata={self.mutates_metadata!r}, mutations_hidden_from_autograd={self.mutations_hidden_from_autograd!r}, mutations_under_no_grad_or_inference_mode={self.mutations_under_no_grad_or_inference_mode!r}, mutation_inductor_storage_resize={self.mutation_inductor_storage_resize!r}, mutates_storage_metadata={self.mutates_storage_metadata!r}, requires_grad={self.requires_grad!r}, keep_input_mutations={self.keep_input_mutations!r})"
 return __repr__