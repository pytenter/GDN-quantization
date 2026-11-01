def __create_fn__():
 def __repr__(self):
  return self.__class__.__qualname__ + f"(group={self.group!r}, cu_seqlens={self.cu_seqlens!r}, cu_seqlens_cpu={self.cu_seqlens_cpu!r}, is_last_rank={self.is_last_rank!r}, pre_num_ranks={self.pre_num_ranks!r}, is_first_rank={self.is_first_rank!r}, post_num_ranks={self.post_num_ranks!r}, conv1d_kernel_size={self.conv1d_kernel_size!r}, pre_num_conv_tokens={self.pre_num_conv_tokens!r})"
 return __repr__