def __create_fn__(_type_onnx_function, _type_op_full_name, _type_is_custom, _type_is_complex, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_is_custom, _dflt_is_complex, _return_type):
 def __init__(self,onnx_function:_type_onnx_function,op_full_name:_type_op_full_name,is_custom:_type_is_custom=_dflt_is_custom,is_complex:_type_is_complex=_dflt_is_complex)->_return_type:
  __dataclass_builtins_object__.__setattr__(self,'onnx_function',onnx_function)
  __dataclass_builtins_object__.__setattr__(self,'op_full_name',op_full_name)
  __dataclass_builtins_object__.__setattr__(self,'is_custom',is_custom)
  __dataclass_builtins_object__.__setattr__(self,'is_complex',is_complex)
 return __init__