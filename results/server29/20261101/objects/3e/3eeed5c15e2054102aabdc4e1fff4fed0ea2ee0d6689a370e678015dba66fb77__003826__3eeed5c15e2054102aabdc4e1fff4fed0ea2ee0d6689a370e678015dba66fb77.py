def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.id,self.type,self.created_at,self.author,self._event,self.content,self.edited,self.hidden,)==(other.id,other.type,other.created_at,other.author,other._event,other.content,other.edited,other.hidden,)
  return NotImplemented
 return __eq__