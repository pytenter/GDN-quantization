def __create_fn__(_type_derived_from, _type_derive_qparams_fn, _type_dtype, _type_quant_min, _type_quant_max, _type_qscheme, _type_ch_axis, _type_is_dynamic, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_quant_min, _dflt_quant_max, _dflt_qscheme, _dflt_ch_axis, _dflt_is_dynamic, _return_type):
 def __init__(self,derived_from:_type_derived_from,derive_qparams_fn:_type_derive_qparams_fn,dtype:_type_dtype,quant_min:_type_quant_min=_dflt_quant_min,quant_max:_type_quant_max=_dflt_quant_max,qscheme:_type_qscheme=_dflt_qscheme,ch_axis:_type_ch_axis=_dflt_ch_axis,is_dynamic:_type_is_dynamic=_dflt_is_dynamic)->_return_type:
  __dataclass_builtins_object__.__setattr__(self,'derived_from',derived_from)
  __dataclass_builtins_object__.__setattr__(self,'derive_qparams_fn',derive_qparams_fn)
  __dataclass_builtins_object__.__setattr__(self,'dtype',dtype)
  __dataclass_builtins_object__.__setattr__(self,'quant_min',quant_min)
  __dataclass_builtins_object__.__setattr__(self,'quant_max',quant_max)
  __dataclass_builtins_object__.__setattr__(self,'qscheme',qscheme)
  __dataclass_builtins_object__.__setattr__(self,'ch_axis',ch_axis)
  __dataclass_builtins_object__.__setattr__(self,'is_dynamic',is_dynamic)
 return __init__