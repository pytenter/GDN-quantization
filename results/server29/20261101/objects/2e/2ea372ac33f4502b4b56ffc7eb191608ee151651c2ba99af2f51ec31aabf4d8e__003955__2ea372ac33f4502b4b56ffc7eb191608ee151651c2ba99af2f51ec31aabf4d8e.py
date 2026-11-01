def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.request_id,self.prompt_ids,self.generated_tokens,self.logprobs,self.error,self.status,self.created_time,)==(other.request_id,other.prompt_ids,other.generated_tokens,other.logprobs,other.error,other.status,other.created_time,)
  return NotImplemented
 return __eq__