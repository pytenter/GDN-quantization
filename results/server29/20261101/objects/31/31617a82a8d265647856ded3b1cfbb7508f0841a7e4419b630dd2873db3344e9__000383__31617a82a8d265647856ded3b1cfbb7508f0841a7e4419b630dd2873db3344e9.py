def __create_fn__(_type__values, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt__values, _return_type):
 def __init__(self,_values:_type__values=_HAS_DEFAULT_FACTORY)->_return_type:
  self._values=_dflt__values() if _values is _HAS_DEFAULT_FACTORY else _values
 return __init__