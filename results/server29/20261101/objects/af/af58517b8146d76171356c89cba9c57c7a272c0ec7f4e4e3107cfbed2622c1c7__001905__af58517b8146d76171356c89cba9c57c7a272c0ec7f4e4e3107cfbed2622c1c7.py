def __create_fn__():
 def __repr__(self):
  return self.__class__.__qualname__ + f"(preferred_execution_providers={self.preferred_execution_providers!r}, infer_execution_providers={self.infer_execution_providers!r}, default_execution_providers={self.default_execution_providers!r}, preallocate_output={self.preallocate_output!r}, use_aot_autograd={self.use_aot_autograd!r}, export_options={self.export_options!r}, ort_session_options={self.ort_session_options!r}, pre_ort_model_transforms={self.pre_ort_model_transforms!r})"
 return __repr__