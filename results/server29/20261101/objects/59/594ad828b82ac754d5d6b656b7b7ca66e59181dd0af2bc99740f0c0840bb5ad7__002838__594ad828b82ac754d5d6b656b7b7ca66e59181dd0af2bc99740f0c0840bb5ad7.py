def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.reads,self.writes,self.index_exprs,self.range_vars,self.var_ranges,)==(other.reads,other.writes,other.index_exprs,other.range_vars,other.var_ranges,)
  return NotImplemented
 return __eq__