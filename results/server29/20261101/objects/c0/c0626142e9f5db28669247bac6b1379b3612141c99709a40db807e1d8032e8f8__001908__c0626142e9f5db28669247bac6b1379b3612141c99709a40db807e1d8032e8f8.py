def __create_fn__(cls, FrozenInstanceError):
 def __delattr__(self,name):
  if type(self) is cls or name in ('preferred_execution_providers','infer_execution_providers','default_execution_providers','preallocate_output','use_aot_autograd','export_options','ort_session_options','pre_ort_model_transforms',):
   raise FrozenInstanceError(f"cannot delete field {name!r}")
  super(cls, self).__delattr__(name)
 return __delattr__