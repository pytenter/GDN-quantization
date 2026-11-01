def __create_fn__(_type_reads, _type_writes, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_reads, _dflt_writes, _return_type):
 def __init__(self,reads:_type_reads=_HAS_DEFAULT_FACTORY,writes:_type_writes=_HAS_DEFAULT_FACTORY)->_return_type:
  self.reads=_dflt_reads() if reads is _HAS_DEFAULT_FACTORY else reads
  self.writes=_dflt_writes() if writes is _HAS_DEFAULT_FACTORY else writes
 return __init__