def __create_fn__():
 def __repr__(self):
  return self.__class__.__qualname__ + f"(request_id={self.request_id!r}, prompt_ids={self.prompt_ids!r}, generated_tokens={self.generated_tokens!r}, logprobs={self.logprobs!r}, error={self.error!r}, status={self.status!r}, created_time={self.created_time!r})"
 return __repr__