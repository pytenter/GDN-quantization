def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.id,self.created_at,self.job_spec,self.schedule,self.suspend,self.concurrency,self.status,self.owner,)==(other.id,other.created_at,other.job_spec,other.schedule,other.suspend,other.concurrency,other.status,other.owner,)
  return NotImplemented
 return __eq__