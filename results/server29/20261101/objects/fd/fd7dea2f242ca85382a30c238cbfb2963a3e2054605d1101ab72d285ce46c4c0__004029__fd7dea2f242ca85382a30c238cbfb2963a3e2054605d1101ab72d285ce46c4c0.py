def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.tp_size,self.torch_device_mesh,)==(other.tp_size,other.torch_device_mesh,)
  return NotImplemented
 return __eq__