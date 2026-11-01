def __create_fn__(_type_showlocals, _type_style, _type_abspath, _type_tbfilter, _type_funcargs, _type_truncate_locals, _type_truncate_args, _type_chain, _type_astcache, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_showlocals, _dflt_style, _dflt_abspath, _dflt_tbfilter, _dflt_funcargs, _dflt_truncate_locals, _dflt_truncate_args, _dflt_chain, _dflt_astcache, _return_type):
 def __init__(self,showlocals:_type_showlocals=_dflt_showlocals,style:_type_style=_dflt_style,abspath:_type_abspath=_dflt_abspath,tbfilter:_type_tbfilter=_dflt_tbfilter,funcargs:_type_funcargs=_dflt_funcargs,truncate_locals:_type_truncate_locals=_dflt_truncate_locals,truncate_args:_type_truncate_args=_dflt_truncate_args,chain:_type_chain=_dflt_chain)->_return_type:
  self.showlocals=showlocals
  self.style=style
  self.abspath=abspath
  self.tbfilter=tbfilter
  self.funcargs=funcargs
  self.truncate_locals=truncate_locals
  self.truncate_args=truncate_args
  self.chain=chain
  self.astcache=_dflt_astcache()
 return __init__