def __create_fn__(_type_name, _type_cpp, _type_triton, _type_cppvec, _type_type_promotion_kind, _type_halide, _type_mps, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_triton, _dflt_cppvec, _dflt_type_promotion_kind, _dflt_halide, _dflt_mps, _return_type):
 def __init__(self,name:_type_name,cpp:_type_cpp,triton:_type_triton=_dflt_triton,cppvec:_type_cppvec=_dflt_cppvec,type_promotion_kind:_type_type_promotion_kind=_dflt_type_promotion_kind,halide:_type_halide=_dflt_halide,mps:_type_mps=_dflt_mps)->_return_type:
  self.name=name
  self.cpp=cpp
  self.triton=triton
  self.cppvec=cppvec
  self.type_promotion_kind=type_promotion_kind
  self.halide=halide
  self.mps=mps
 return __init__