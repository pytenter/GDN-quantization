def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.args,self.plugins,self.dir,)==(other.args,other.plugins,other.dir,)
  return NotImplemented
 return __eq__