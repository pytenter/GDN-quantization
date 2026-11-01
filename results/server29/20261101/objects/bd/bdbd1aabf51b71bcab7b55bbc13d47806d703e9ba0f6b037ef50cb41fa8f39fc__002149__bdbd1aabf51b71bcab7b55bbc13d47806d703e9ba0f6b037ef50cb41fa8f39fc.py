def __create_fn__(_type_entries, _type_index, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_entries, _dflt_index, _return_type):
 def __init__(self,entries:_type_entries=_HAS_DEFAULT_FACTORY,index:_type_index=_dflt_index)->_return_type:
  self.entries=_dflt_entries() if entries is _HAS_DEFAULT_FACTORY else entries
  self.index=index
 return __init__