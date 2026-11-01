def __create_fn__(cls, FrozenInstanceError):
 def __setattr__(self,name,value):
  if type(self) is cls or name in ('id','ndim','dtype','device','size','dynamo_dynamic_indices','layout','is_inference','is_leaf','requires_grad','is_sparse','is_mkldnn','is_functorch_wrapped','is_batchedtensor','is_legacy_batchedtensor','is_gradtrackingtensor','is_view','is_nested','nested_int','is_traceable_wrapper_subclass','is_functional','is_conj','is_neg','is_parameter','stride','storage_offset','storage','sparse_dim','dense_dim','is_coalesced','crow_indices','col_indices','ccol_indices','row_indices','values','unwrapped','bdim','base','attrs','creation_meta','grad','ctx','type','fake_mode','view_func','level','current_level','functorch_stack','autograd_meta_from','data',):
   raise FrozenInstanceError(f"cannot assign to field {name!r}")
  super(cls, self).__setattr__(name, value)
 return __setattr__