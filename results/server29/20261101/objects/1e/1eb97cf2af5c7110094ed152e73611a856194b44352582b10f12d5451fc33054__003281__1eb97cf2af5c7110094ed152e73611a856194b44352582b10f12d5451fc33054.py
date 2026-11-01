def __create_fn__(_type_input_dtype, _type_input2_dtype, _type_output_dtype, _type_compute_dtype, _type_vec_isa_cls, _type_register_blocking, _type_extra_check, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_extra_check, _return_type):
 def __init__(self,input_dtype:_type_input_dtype,input2_dtype:_type_input2_dtype,output_dtype:_type_output_dtype,compute_dtype:_type_compute_dtype,vec_isa_cls:_type_vec_isa_cls,register_blocking:_type_register_blocking,extra_check:_type_extra_check=_dflt_extra_check)->_return_type:
  self.input_dtype=input_dtype
  self.input2_dtype=input2_dtype
  self.output_dtype=output_dtype
  self.compute_dtype=compute_dtype
  self.vec_isa_cls=vec_isa_cls
  self.register_blocking=register_blocking
  self.extra_check=extra_check
 return __init__