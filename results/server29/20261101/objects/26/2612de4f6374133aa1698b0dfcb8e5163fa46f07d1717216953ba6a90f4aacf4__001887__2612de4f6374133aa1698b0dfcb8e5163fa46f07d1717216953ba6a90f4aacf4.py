def __create_fn__(_type_cleanup_fn, _type_proxy, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_cleanup_fn, _dflt_proxy, _return_type):
 def __init__(self,cleanup_fn:_type_cleanup_fn=_dflt_cleanup_fn,proxy:_type_proxy=_dflt_proxy)->_return_type:
  self.cleanup_fn=cleanup_fn
  self.proxy=proxy
 return __init__