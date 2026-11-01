def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.id,self.type,self.created_at,self.author,self._event,self.new_status,)==(other.id,other.type,other.created_at,other.author,other._event,other.new_status,)
  return NotImplemented
 return __eq__