def __create_fn__(_type_target, _type_description, _type_kinds, _type_properties, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_description, _dflt_kinds, _dflt_properties, _return_type):
 def __init__(self,target:_type_target,description:_type_description=_dflt_description,kinds:_type_kinds=_HAS_DEFAULT_FACTORY,properties:_type_properties=_dflt_properties)->_return_type:
  self.target=target
  self.description=description
  self.kinds=_dflt_kinds() if kinds is _HAS_DEFAULT_FACTORY else kinds
  self.properties=properties
 return __init__