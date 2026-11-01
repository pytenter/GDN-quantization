def __create_fn__(_type_data, _type_layout, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _return_type):
 def __init__(self,*,data:_type_data,layout:_type_layout)->_return_type:
  __dataclass_builtins_object__.__setattr__(self,'data',data)
  __dataclass_builtins_object__.__setattr__(self,'layout',layout)
  self.__post_init__()
 return __init__