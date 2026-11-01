def __create_fn__(_type_var, _type_size, _type_offset, _type_tiled_size, _type_steps, _type_parallel, _type_simd_omp, _type_simd_vec, _type_collapsed, _type_is_reduction, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_var, _dflt_size, _dflt_offset, _dflt_tiled_size, _dflt_steps, _dflt_parallel, _dflt_simd_omp, _dflt_simd_vec, _dflt_collapsed, _dflt_is_reduction, _return_type):
 def __init__(self,var:_type_var=_dflt_var,size:_type_size=_dflt_size,offset:_type_offset=_dflt_offset,tiled_size:_type_tiled_size=_dflt_tiled_size,steps:_type_steps=_dflt_steps,parallel:_type_parallel=_dflt_parallel,simd_omp:_type_simd_omp=_dflt_simd_omp,simd_vec:_type_simd_vec=_dflt_simd_vec,collapsed:_type_collapsed=_dflt_collapsed,is_reduction:_type_is_reduction=_dflt_is_reduction)->_return_type:
  self.var=var
  self.size=size
  self.offset=offset
  self.tiled_size=tiled_size
  self.steps=steps
  self.parallel=parallel
  self.simd_omp=simd_omp
  self.simd_vec=simd_vec
  self.collapsed=collapsed
  self.is_reduction=is_reduction
  self.__post_init__()
 return __init__