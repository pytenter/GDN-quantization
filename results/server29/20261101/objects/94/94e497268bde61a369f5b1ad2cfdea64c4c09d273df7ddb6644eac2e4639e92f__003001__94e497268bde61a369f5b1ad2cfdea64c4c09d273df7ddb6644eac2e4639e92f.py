def __create_fn__(_type_name, _type_layout, _type_data, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _return_type):
 def __init__(self,*,name:_type_name,layout:_type_layout,data:_type_data)->_return_type:
  self.name=name
  self.layout=layout
  self.data=data
  self.__post_init__()
 return __init__