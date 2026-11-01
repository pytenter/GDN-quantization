def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.tool,self.analysis_tool_log_files,self.invocation,self.properties,)==(other.tool,other.analysis_tool_log_files,other.invocation,other.properties,)
  return NotImplemented
 return __eq__