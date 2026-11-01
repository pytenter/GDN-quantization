def __create_fn__(_type_reads, _type_writes, _type_index_exprs, _type_range_vars, _type_var_ranges, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_range_vars, _dflt_var_ranges, _return_type):
 def __init__(self,reads:_type_reads,writes:_type_writes,index_exprs:_type_index_exprs,range_vars:_type_range_vars=_dflt_range_vars,var_ranges:_type_var_ranges=_dflt_var_ranges)->_return_type:
  self.reads=reads
  self.writes=writes
  self.index_exprs=index_exprs
  self.range_vars=range_vars
  self.var_ranges=var_ranges
 return __init__