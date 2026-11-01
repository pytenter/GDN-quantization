def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.name,self.version,self.options,self.diagnostics,self._inflight_diagnostics,self._previous_log_level,self.logger,self._bound_diagnostic_type,)==(other.name,other.version,other.options,other.diagnostics,other._inflight_diagnostics,other._previous_log_level,other.logger,other._bound_diagnostic_type,)
  return NotImplemented
 return __eq__