def __create_fn__(_type__exc_stack, _type__current_exception, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt__exc_stack, _dflt__current_exception, _return_type):
 def __init__(self,_exc_stack:_type__exc_stack=_HAS_DEFAULT_FACTORY,_current_exception:_type__current_exception=_dflt__current_exception)->_return_type:
  self._exc_stack=_dflt__exc_stack() if _exc_stack is _HAS_DEFAULT_FACTORY else _exc_stack
  self._current_exception=_current_exception
 return __init__