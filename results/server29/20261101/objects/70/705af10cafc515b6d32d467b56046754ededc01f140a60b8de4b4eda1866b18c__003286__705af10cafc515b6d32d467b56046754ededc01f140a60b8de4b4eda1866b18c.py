def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.architectures,self.cuda_version,self.instantiation_level,self.kernel_filter_file,self.selected_kernel_list,self.interface_dir,)==(other.architectures,other.cuda_version,other.instantiation_level,other.kernel_filter_file,other.selected_kernel_list,other.interface_dir,)
  return NotImplemented
 return __eq__