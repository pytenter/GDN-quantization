def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.input_qspec_map,self.output_qspec,self.allow_implicit_sharing,self._annotated,)==(other.input_qspec_map,other.output_qspec,other.allow_implicit_sharing,other._annotated,)
  return NotImplemented
 return __eq__