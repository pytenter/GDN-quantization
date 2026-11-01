def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.count_per_thread_contiguous,self.count_per_thread_broadcast,self.count_per_thread_non_contiguous,self.bytes_per_thread_contiguous,self.bytes_per_thread_broadcast,self.bytes_per_thread_non_contiguous,self.bytes_contiguous_or_broadcast,self.bytes_non_contiguous,)==(other.count_per_thread_contiguous,other.count_per_thread_broadcast,other.count_per_thread_non_contiguous,other.bytes_per_thread_contiguous,other.bytes_per_thread_broadcast,other.bytes_per_thread_non_contiguous,other.bytes_contiguous_or_broadcast,other.bytes_non_contiguous,)
  return NotImplemented
 return __eq__