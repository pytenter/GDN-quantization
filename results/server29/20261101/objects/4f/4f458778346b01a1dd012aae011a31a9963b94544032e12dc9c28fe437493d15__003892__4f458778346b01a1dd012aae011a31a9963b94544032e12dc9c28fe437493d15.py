def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.item_object_id,self.item_id,self.item_type,self.position,self.note,)==(other.item_object_id,other.item_id,other.item_type,other.position,other.note,)
  return NotImplemented
 return __eq__