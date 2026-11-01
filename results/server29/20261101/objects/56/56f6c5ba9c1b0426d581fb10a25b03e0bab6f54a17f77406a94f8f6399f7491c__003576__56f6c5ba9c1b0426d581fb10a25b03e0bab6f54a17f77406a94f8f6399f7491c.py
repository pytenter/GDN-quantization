def __create_fn__(_type_scope, _type_params, _type_autouse, _type_ids, _type_name, _type__ispytest, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_autouse, _dflt_ids, _dflt_name, _dflt__ispytest, _return_type):
 def __init__(self,scope:_type_scope,params:_type_params,autouse:_type_autouse=_dflt_autouse,ids:_type_ids=_dflt_ids,name:_type_name=_dflt_name,_ispytest:_type__ispytest=_dflt__ispytest)->_return_type:
  __dataclass_builtins_object__.__setattr__(self,'scope',scope)
  __dataclass_builtins_object__.__setattr__(self,'params',params)
  __dataclass_builtins_object__.__setattr__(self,'autouse',autouse)
  __dataclass_builtins_object__.__setattr__(self,'ids',ids)
  __dataclass_builtins_object__.__setattr__(self,'name',name)
  self.__post_init__(_ispytest)
 return __init__