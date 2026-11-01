def __create_fn__():
 def __repr__(self):
  return self.__class__.__qualname__ + f"(generated_kernel_count={self.generated_kernel_count!r}, generated_cpp_vec_kernel_count={self.generated_cpp_vec_kernel_count!r}, ir_nodes_pre_fusion={self.ir_nodes_pre_fusion!r}, cpp_to_dtype_count={self.cpp_to_dtype_count!r}, num_bytes_accessed={self.num_bytes_accessed!r}, num_matches_for_scatter_upon_const_tensor={self.num_matches_for_scatter_upon_const_tensor!r})"
 return __repr__