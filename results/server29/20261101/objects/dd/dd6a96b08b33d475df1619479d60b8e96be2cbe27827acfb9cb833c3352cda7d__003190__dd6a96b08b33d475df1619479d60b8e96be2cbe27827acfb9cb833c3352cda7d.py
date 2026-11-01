def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.inductor_meta,self.mode,self.prefix,self.x_grid,self.y_grid,self.z_grid,)==(other.inductor_meta,other.mode,other.prefix,other.x_grid,other.y_grid,other.z_grid,)
  return NotImplemented
 return __eq__