def __create_fn__(_type_code, _type_closure, _type_globals, _type_locals, _type_builtins, _type_code_options, _type_name_to_modrec, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_globals, _dflt_locals, _dflt_builtins, _dflt_code_options, _dflt_name_to_modrec, _return_type):
 def __init__(self,code:_type_code,closure:_type_closure,globals:_type_globals=_HAS_DEFAULT_FACTORY,locals:_type_locals=_HAS_DEFAULT_FACTORY,builtins:_type_builtins=_HAS_DEFAULT_FACTORY,code_options:_type_code_options=_HAS_DEFAULT_FACTORY,name_to_modrec:_type_name_to_modrec=_HAS_DEFAULT_FACTORY)->_return_type:
  self.code=code
  self.closure=closure
  self.globals=_dflt_globals() if globals is _HAS_DEFAULT_FACTORY else globals
  self.locals=_dflt_locals() if locals is _HAS_DEFAULT_FACTORY else locals
  self.builtins=_dflt_builtins() if builtins is _HAS_DEFAULT_FACTORY else builtins
  self.code_options=_dflt_code_options() if code_options is _HAS_DEFAULT_FACTORY else code_options
  self.name_to_modrec=_dflt_name_to_modrec() if name_to_modrec is _HAS_DEFAULT_FACTORY else name_to_modrec
 return __init__