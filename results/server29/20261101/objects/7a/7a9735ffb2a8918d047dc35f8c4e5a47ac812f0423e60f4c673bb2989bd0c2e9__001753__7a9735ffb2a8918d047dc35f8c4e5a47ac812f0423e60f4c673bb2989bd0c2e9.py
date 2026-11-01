def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.output_spec,self.redistribute_schema,self.needs_redistribute,)==(other.output_spec,other.redistribute_schema,other.needs_redistribute,)
  return NotImplemented
 return __eq__