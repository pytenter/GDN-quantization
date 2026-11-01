def __create_fn__(_type_out_metas, _type_fwd_output_strides, _type_needs_post_compile, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_out_metas, _dflt_fwd_output_strides, _dflt_needs_post_compile, _return_type):
 def __init__(self,out_metas:_type_out_metas=_HAS_DEFAULT_FACTORY,fwd_output_strides:_type_fwd_output_strides=_dflt_fwd_output_strides,needs_post_compile:_type_needs_post_compile=_dflt_needs_post_compile)->_return_type:
  self.out_metas=_dflt_out_metas() if out_metas is _HAS_DEFAULT_FACTORY else out_metas
  self.fwd_output_strides=fwd_output_strides
  self.needs_post_compile=needs_post_compile
 return __init__