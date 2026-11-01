def __create_fn__(_type_name, _type_fn_call_name, _type_args, _type_ret, _type_sub_idx, _type_is_pure, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_sub_idx, _dflt_is_pure, _return_type):
 def __init__(self,name:_type_name,fn_call_name:_type_fn_call_name,args:_type_args,ret:_type_ret,sub_idx:_type_sub_idx=_dflt_sub_idx,is_pure:_type_is_pure=_dflt_is_pure)->_return_type:
  __dataclass_builtins_object__.__setattr__(self,'name',name)
  __dataclass_builtins_object__.__setattr__(self,'fn_call_name',fn_call_name)
  __dataclass_builtins_object__.__setattr__(self,'args',args)
  __dataclass_builtins_object__.__setattr__(self,'ret',ret)
  __dataclass_builtins_object__.__setattr__(self,'sub_idx',sub_idx)
  __dataclass_builtins_object__.__setattr__(self,'is_pure',is_pure)
  self.__post_init__()
 return __init__