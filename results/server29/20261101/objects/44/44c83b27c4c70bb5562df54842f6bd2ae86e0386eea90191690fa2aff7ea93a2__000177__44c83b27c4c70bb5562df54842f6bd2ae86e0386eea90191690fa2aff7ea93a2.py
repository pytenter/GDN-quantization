def __create_fn__(_type_view, _type_view_copy, _type_view_inplace, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _return_type):
 def __init__(self,view:_type_view,view_copy:_type_view_copy,view_inplace:_type_view_inplace)->_return_type:
  __dataclass_builtins_object__.__setattr__(self,'view',view)
  __dataclass_builtins_object__.__setattr__(self,'view_copy',view_copy)
  __dataclass_builtins_object__.__setattr__(self,'view_inplace',view_inplace)
  self.__post_init__()
 return __init__