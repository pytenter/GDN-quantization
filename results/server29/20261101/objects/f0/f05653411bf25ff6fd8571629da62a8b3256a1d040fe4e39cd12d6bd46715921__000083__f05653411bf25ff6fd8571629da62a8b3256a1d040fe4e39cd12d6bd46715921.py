def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.name,self.supported_dtypes,self.ufunc_key,)==(other.name,other.supported_dtypes,other.ufunc_key,)
  return NotImplemented
 return __eq__