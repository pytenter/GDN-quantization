def __create_fn__():
 def __hash__(self):
  return hash((self.kernel_name,self.triton_version,self.autotune_entries,self.default_config,))
 return __hash__