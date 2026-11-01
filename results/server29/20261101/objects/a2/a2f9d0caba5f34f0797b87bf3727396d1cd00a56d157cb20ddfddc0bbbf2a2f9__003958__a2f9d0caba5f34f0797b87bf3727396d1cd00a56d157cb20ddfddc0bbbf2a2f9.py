def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.request_id,self.full_prompt_ids,self.prompt_ids,self.remaining_prompt_ids,self.static_outputs,self.allocated_blocks,self.position_offset,self._status,self.max_new_tokens,self.eos_token_id,self.created_time,self.error,self.lifespan,)==(other.request_id,other.full_prompt_ids,other.prompt_ids,other.remaining_prompt_ids,other.static_outputs,other.allocated_blocks,other.position_offset,other._status,other.max_new_tokens,other.eos_token_id,other.created_time,other.error,other.lifespan,)
  return NotImplemented
 return __eq__