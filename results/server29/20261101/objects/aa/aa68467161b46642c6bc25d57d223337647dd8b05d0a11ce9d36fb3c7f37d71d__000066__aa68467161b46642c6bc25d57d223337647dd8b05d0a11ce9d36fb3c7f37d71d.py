def __create_fn__(cls, FrozenInstanceError):
 def __setattr__(self,name,value):
  if type(self) is cls or name in ('namespace','func','use_const_ref_for_mutable_tensors','device_guard','device_check','python_module','category_override','variants','manual_kernel_registration','manual_cpp_binding','loc','autogen','ufunc_inner_loop','structured','structured_delegate','structured_inherits','precomputed','cpp_no_default_args','is_abstract','has_composite_implicit_autograd_kernel','has_composite_implicit_autograd_nested_tensor_kernel','has_composite_explicit_autograd_kernel','has_composite_explicit_autograd_non_functional_kernel','tags',):
   raise FrozenInstanceError(f"cannot assign to field {name!r}")
  super(cls, self).__setattr__(name, value)
 return __setattr__