def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.var,self.size,self.offset,self.tiled_size,self.steps,self.parallel,self.simd_omp,self.simd_vec,self.collapsed,self.is_reduction,)==(other.var,other.size,other.offset,other.tiled_size,other.steps,other.parallel,other.simd_omp,other.simd_vec,other.collapsed,other.is_reduction,)
  return NotImplemented
 return __eq__