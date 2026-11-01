def __create_fn__(_type_base, _type_index, _type_index_is_slice, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_index_is_slice, _return_type):
 def __init__(self,base:_type_base,index:_type_index,index_is_slice:_type_index_is_slice=_dflt_index_is_slice)->_return_type:
  __dataclass_builtins_object__.__setattr__(self,'base',base)
  __dataclass_builtins_object__.__setattr__(self,'index',index)
  __dataclass_builtins_object__.__setattr__(self,'index_is_slice',index_is_slice)
  self.__post_init__()
 return __init__