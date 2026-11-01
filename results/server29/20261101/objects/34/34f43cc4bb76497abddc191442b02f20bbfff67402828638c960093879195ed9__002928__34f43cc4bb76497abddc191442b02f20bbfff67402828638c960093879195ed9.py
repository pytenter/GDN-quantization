def __create_fn__(_type_data, _type_size, _type_reindex, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _return_type):
 def __init__(self,*,data:_type_data,size:_type_size,reindex:_type_reindex)->_return_type:
  __dataclass_builtins_object__.__setattr__(self,'data',data)
  __dataclass_builtins_object__.__setattr__(self,'size',size)
  __dataclass_builtins_object__.__setattr__(self,'reindex',reindex)
  self.__post_init__()
 return __init__