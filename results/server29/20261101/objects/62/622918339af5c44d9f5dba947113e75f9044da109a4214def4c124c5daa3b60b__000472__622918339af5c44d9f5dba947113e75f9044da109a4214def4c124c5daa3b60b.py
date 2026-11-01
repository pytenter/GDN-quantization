def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.log_qname_to_level,self.artifact_names,)==(other.log_qname_to_level,other.artifact_names,)
  return NotImplemented
 return __eq__