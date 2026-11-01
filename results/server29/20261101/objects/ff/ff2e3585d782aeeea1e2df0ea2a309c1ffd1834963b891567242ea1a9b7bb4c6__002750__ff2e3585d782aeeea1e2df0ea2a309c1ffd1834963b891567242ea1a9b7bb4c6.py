def __create_fn__():
 def __repr__(self):
  return self.__class__.__qualname__ + f"(library_basename={self.library_basename!r}, input_names={self.input_names!r}, output_names={self.output_names!r}, floating_point_input_dtype={self.floating_point_input_dtype!r}, floating_point_output_dtype={self.floating_point_output_dtype!r}, aot_inductor_model_is_cpu={self.aot_inductor_model_is_cpu!r})"
 return __repr__