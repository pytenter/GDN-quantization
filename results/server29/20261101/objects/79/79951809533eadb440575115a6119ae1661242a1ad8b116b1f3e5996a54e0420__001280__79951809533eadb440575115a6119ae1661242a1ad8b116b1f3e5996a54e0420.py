def __create_fn__(_type_loc, _type_locals, _type_symbols, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_loc, _dflt_locals, _dflt_symbols, _return_type):
 def __init__(self,loc:_type_loc=_dflt_loc,locals:_type_locals=_HAS_DEFAULT_FACTORY,symbols:_type_symbols=_HAS_DEFAULT_FACTORY)->_return_type:
  self.loc=loc
  self.locals=_dflt_locals() if locals is _HAS_DEFAULT_FACTORY else locals
  self.symbols=_dflt_symbols() if symbols is _HAS_DEFAULT_FACTORY else symbols
 return __init__