def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.argnames,self.parametersets,self.idfn,self.ids,self.config,self.nodeid,)==(other.argnames,other.parametersets,other.idfn,other.ids,other.config,other.nodeid,)
  return NotImplemented
 return __eq__