def __create_fn__(_type_params, _type_indices, _type__arg2scope, _type__idlist, _type_marks, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_params, _dflt_indices, _dflt__arg2scope, _dflt__idlist, _dflt_marks, _return_type):
 def __init__(self,params:_type_params=_HAS_DEFAULT_FACTORY,indices:_type_indices=_HAS_DEFAULT_FACTORY,_arg2scope:_type__arg2scope=_HAS_DEFAULT_FACTORY,_idlist:_type__idlist=_HAS_DEFAULT_FACTORY,marks:_type_marks=_HAS_DEFAULT_FACTORY)->_return_type:
  __dataclass_builtins_object__.__setattr__(self,'params',_dflt_params() if params is _HAS_DEFAULT_FACTORY else params)
  __dataclass_builtins_object__.__setattr__(self,'indices',_dflt_indices() if indices is _HAS_DEFAULT_FACTORY else indices)
  __dataclass_builtins_object__.__setattr__(self,'_arg2scope',_dflt__arg2scope() if _arg2scope is _HAS_DEFAULT_FACTORY else _arg2scope)
  __dataclass_builtins_object__.__setattr__(self,'_idlist',_dflt__idlist() if _idlist is _HAS_DEFAULT_FACTORY else _idlist)
  __dataclass_builtins_object__.__setattr__(self,'marks',_dflt_marks() if marks is _HAS_DEFAULT_FACTORY else marks)
 return __init__