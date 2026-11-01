def __create_fn__(_type_request_id, _type_prompt_ids, _type_generated_tokens, _type_logprobs, _type_error, _type_status, _type_created_time, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_prompt_ids, _dflt_generated_tokens, _dflt_logprobs, _dflt_error, _dflt_status, _dflt_created_time, _return_type):
 def __init__(self,request_id:_type_request_id,prompt_ids:_type_prompt_ids=_HAS_DEFAULT_FACTORY,generated_tokens:_type_generated_tokens=_HAS_DEFAULT_FACTORY,logprobs:_type_logprobs=_HAS_DEFAULT_FACTORY,error:_type_error=_dflt_error,status:_type_status=_dflt_status,created_time:_type_created_time=_HAS_DEFAULT_FACTORY)->_return_type:
  self.request_id=request_id
  self.prompt_ids=_dflt_prompt_ids() if prompt_ids is _HAS_DEFAULT_FACTORY else prompt_ids
  self.generated_tokens=_dflt_generated_tokens() if generated_tokens is _HAS_DEFAULT_FACTORY else generated_tokens
  self.logprobs=_dflt_logprobs() if logprobs is _HAS_DEFAULT_FACTORY else logprobs
  self.error=error
  self.status=status
  self.created_time=_dflt_created_time() if created_time is _HAS_DEFAULT_FACTORY else created_time
 return __init__