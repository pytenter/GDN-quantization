def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.op,self.args_schema,self.kwargs_schema,self.schema_info,)==(other.op,other.args_schema,other.kwargs_schema,other.schema_info,)
  return NotImplemented
 return __eq__