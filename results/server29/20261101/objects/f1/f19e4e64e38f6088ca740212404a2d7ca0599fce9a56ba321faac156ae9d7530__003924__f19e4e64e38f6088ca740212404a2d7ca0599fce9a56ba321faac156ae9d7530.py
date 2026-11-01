def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.target_class,self.index,self.layer_name,self.class_name,)==(other.target_class,other.index,other.layer_name,other.class_name,)
  return NotImplemented
 return __eq__