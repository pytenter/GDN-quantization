def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.name,self.cpp,self.triton,self.cppvec,self.type_promotion_kind,self.halide,self.mps,)==(other.name,other.cpp,other.triton,other.cppvec,other.type_promotion_kind,other.halide,other.mps,)
  return NotImplemented
 return __eq__