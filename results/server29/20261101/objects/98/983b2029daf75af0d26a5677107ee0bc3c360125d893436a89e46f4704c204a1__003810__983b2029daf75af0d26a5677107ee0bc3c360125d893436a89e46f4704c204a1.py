def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.last_job,self.next_job_run_at,)==(other.last_job,other.next_job_run_at,)
  return NotImplemented
 return __eq__