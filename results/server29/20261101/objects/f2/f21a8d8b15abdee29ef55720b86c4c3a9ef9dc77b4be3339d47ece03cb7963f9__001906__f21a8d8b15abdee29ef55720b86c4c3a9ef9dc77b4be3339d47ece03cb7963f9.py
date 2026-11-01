def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.preferred_execution_providers,self.infer_execution_providers,self.default_execution_providers,self.preallocate_output,self.use_aot_autograd,self.export_options,self.ort_session_options,self.pre_ort_model_transforms,)==(other.preferred_execution_providers,other.infer_execution_providers,other.default_execution_providers,other.preallocate_output,other.use_aot_autograd,other.export_options,other.ort_session_options,other.pre_ort_model_transforms,)
  return NotImplemented
 return __eq__