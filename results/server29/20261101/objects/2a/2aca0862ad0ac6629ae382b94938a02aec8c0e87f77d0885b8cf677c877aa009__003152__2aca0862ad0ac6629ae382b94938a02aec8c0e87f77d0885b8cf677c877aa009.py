def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.device,self.process,self.request_queue,self.response_queue,)==(other.device,other.process,other.request_queue,other.response_queue,)
  return NotImplemented
 return __eq__