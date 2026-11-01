def __create_fn__(_type_name, _type_state, _type_token, _type_pid, _type_hostname, _type_start, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_start, _return_type):
 def __init__(self,name:_type_name,state:_type_state,token:_type_token,pid:_type_pid,hostname:_type_hostname,start:_type_start=_dflt_start)->_return_type:
  __dataclass_builtins_object__.__setattr__(self,'name',name)
  __dataclass_builtins_object__.__setattr__(self,'state',state)
  __dataclass_builtins_object__.__setattr__(self,'token',token)
  __dataclass_builtins_object__.__setattr__(self,'pid',pid)
  __dataclass_builtins_object__.__setattr__(self,'hostname',hostname)
  __dataclass_builtins_object__.__setattr__(self,'start',start)
 return __init__