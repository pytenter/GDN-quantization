def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self._rule_id_name_set,)==(other._rule_id_name_set,)
  return NotImplemented
 return __eq__