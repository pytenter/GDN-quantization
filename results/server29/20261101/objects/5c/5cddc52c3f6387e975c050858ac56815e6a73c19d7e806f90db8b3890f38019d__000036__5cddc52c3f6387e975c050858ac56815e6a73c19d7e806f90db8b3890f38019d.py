def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.precision,self.threshold,self.edgeitems,self.linewidth,self.sci_mode,)==(other.precision,other.threshold,other.edgeitems,other.linewidth,other.sci_mode,)
  return NotImplemented
 return __eq__