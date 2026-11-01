def __create_fn__(_type_guard_export_fn, _type_guard_fail_fn, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_guard_export_fn, _dflt_guard_fail_fn, _return_type):
 def __init__(self,guard_export_fn:_type_guard_export_fn=_dflt_guard_export_fn,guard_fail_fn:_type_guard_fail_fn=_dflt_guard_fail_fn)->_return_type:
  self.guard_export_fn=guard_export_fn
  self.guard_fail_fn=guard_fail_fn
 return __init__