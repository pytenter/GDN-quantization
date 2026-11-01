def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.activities,self.schedule_option,self.on_trace_ready,self.record_shapes,self.profile_memory,self.with_stack,self.with_flops,self.with_modules,self.output_trace_dir,)==(other.activities,other.schedule_option,other.on_trace_ready,other.record_shapes,other.profile_memory,other.with_stack,other.with_flops,other.with_modules,other.output_trace_dir,)
  return NotImplemented
 return __eq__