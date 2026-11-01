def __create_fn__(_type_module, _type_accessed_attrs, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_accessed_attrs, _return_type):
 def __init__(self,module:_type_module,accessed_attrs:_type_accessed_attrs=_HAS_DEFAULT_FACTORY)->_return_type:
  self.module=module
  self.accessed_attrs=_dflt_accessed_attrs() if accessed_attrs is _HAS_DEFAULT_FACTORY else accessed_attrs
 return __init__