def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.fake_mode,self.state_dict_paths,)==(other.fake_mode,other.state_dict_paths,)
  return NotImplemented
 return __eq__