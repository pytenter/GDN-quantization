def __create_fn__(_type_path, _type_lineno, _type_message, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _return_type):
 def __init__(self,path:_type_path,lineno:_type_lineno,message:_type_message)->_return_type:
  self.path=path
  self.lineno=lineno
  self.message=message
  self.__post_init__()
 return __init__