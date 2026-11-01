def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.showlocals,self.style,self.abspath,self.tbfilter,self.funcargs,self.truncate_locals,self.truncate_args,self.chain,self.astcache,)==(other.showlocals,other.style,other.abspath,other.tbfilter,other.funcargs,other.truncate_locals,other.truncate_args,other.chain,other.astcache,)
  return NotImplemented
 return __eq__