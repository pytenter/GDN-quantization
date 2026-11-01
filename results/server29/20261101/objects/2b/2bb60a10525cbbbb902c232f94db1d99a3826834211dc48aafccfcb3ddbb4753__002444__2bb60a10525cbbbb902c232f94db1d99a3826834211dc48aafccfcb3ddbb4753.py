def __create_fn__(cls, FrozenInstanceError):
 def __setattr__(self,name,value):
  if type(self) is cls or name in ('is_leaf','mutates_data','mutates_metadata','mutations_hidden_from_autograd','mutations_under_no_grad_or_inference_mode','mutation_inductor_storage_resize','mutates_storage_metadata','requires_grad','keep_input_mutations',):
   raise FrozenInstanceError(f"cannot assign to field {name!r}")
  super(cls, self).__setattr__(name, value)
 return __setattr__