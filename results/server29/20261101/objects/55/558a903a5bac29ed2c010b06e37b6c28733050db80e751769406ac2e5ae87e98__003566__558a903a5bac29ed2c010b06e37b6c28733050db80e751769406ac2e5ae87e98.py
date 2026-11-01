def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.argname,self.param_index,self.scoped_item_path,self.item_cls,)==(other.argname,other.param_index,other.scoped_item_path,other.item_cls,)
  return NotImplemented
 return __eq__