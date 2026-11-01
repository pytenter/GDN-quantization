def __create_fn__():
 def __repr__(self):
  return self.__class__.__qualname__ + f"(request_id={self.request_id!r}, full_prompt_ids={self.full_prompt_ids!r}, prompt_ids={self.prompt_ids!r}, remaining_prompt_ids={self.remaining_prompt_ids!r}, static_outputs={self.static_outputs!r}, allocated_blocks={self.allocated_blocks!r}, position_offset={self.position_offset!r}, _status={self._status!r}, max_new_tokens={self.max_new_tokens!r}, eos_token_id={self.eos_token_id!r}, created_time={self.created_time!r}, error={self.error!r}, lifespan={self.lifespan!r})"
 return __repr__