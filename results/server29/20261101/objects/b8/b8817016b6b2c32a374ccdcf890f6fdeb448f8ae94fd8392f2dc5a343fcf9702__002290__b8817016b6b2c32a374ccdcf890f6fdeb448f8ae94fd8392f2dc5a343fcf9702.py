def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.generated_kernel_count,self.generated_cpp_vec_kernel_count,self.ir_nodes_pre_fusion,self.cpp_to_dtype_count,self.num_bytes_accessed,self.num_matches_for_scatter_upon_const_tensor,)==(other.generated_kernel_count,other.generated_cpp_vec_kernel_count,other.ir_nodes_pre_fusion,other.cpp_to_dtype_count,other.num_bytes_accessed,other.num_matches_for_scatter_upon_const_tensor,)
  return NotImplemented
 return __eq__