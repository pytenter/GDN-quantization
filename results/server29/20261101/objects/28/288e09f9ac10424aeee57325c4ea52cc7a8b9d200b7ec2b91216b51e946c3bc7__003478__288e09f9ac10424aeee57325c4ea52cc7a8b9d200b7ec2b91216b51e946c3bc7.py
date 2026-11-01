def __create_fn__(_type_reprtraceback, _type_reprcrash, _type_sections, _type_chain, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_sections, _return_type):
 def __init__(self,reprtraceback:_type_reprtraceback,reprcrash:_type_reprcrash,chain:_type_chain)->_return_type:
  self.reprtraceback=reprtraceback
  self.reprcrash=reprcrash
  self.sections=_dflt_sections()
  self.chain=chain
 return __init__