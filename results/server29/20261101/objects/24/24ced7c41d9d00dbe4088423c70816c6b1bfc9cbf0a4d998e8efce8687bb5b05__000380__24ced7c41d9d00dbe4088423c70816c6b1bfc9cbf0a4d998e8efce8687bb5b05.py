def __create_fn__(_type_by_id, _type_by_key, _type_by_version, _type__by_id_keyset, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_by_id, _dflt_by_key, _dflt_by_version, _dflt__by_id_keyset, _return_type):
 def __init__(self,by_id:_type_by_id=_dflt_by_id,by_key:_type_by_key=_HAS_DEFAULT_FACTORY,by_version:_type_by_version=_HAS_DEFAULT_FACTORY,_by_id_keyset:_type__by_id_keyset=_HAS_DEFAULT_FACTORY)->_return_type:
  self.by_id=by_id
  self.by_key=_dflt_by_key() if by_key is _HAS_DEFAULT_FACTORY else by_key
  self.by_version=_dflt_by_version() if by_version is _HAS_DEFAULT_FACTORY else by_version
  self._by_id_keyset=_dflt__by_id_keyset() if _by_id_keyset is _HAS_DEFAULT_FACTORY else _by_id_keyset
 return __init__