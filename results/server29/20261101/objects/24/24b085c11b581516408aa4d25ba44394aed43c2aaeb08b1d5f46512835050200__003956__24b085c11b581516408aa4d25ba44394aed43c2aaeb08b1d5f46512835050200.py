def __create_fn__(_type_request_id, _type_full_prompt_ids, _type_prompt_ids, _type_remaining_prompt_ids, _type_static_outputs, _type_allocated_blocks, _type_position_offset, _type__status, _type_max_new_tokens, _type_eos_token_id, _type_created_time, _type_error, _type_lifespan, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_full_prompt_ids, _dflt_prompt_ids, _dflt_remaining_prompt_ids, _dflt_static_outputs, _dflt_allocated_blocks, _dflt_position_offset, _dflt__status, _dflt_max_new_tokens, _dflt_eos_token_id, _dflt_created_time, _dflt_error, _dflt_lifespan, _return_type):
 def __init__(self,request_id:_type_request_id,full_prompt_ids:_type_full_prompt_ids=_dflt_full_prompt_ids,prompt_ids:_type_prompt_ids=_dflt_prompt_ids,remaining_prompt_ids:_type_remaining_prompt_ids=_HAS_DEFAULT_FACTORY,static_outputs:_type_static_outputs=_HAS_DEFAULT_FACTORY,allocated_blocks:_type_allocated_blocks=_dflt_allocated_blocks,position_offset:_type_position_offset=_dflt_position_offset,_status:_type__status=_dflt__status,max_new_tokens:_type_max_new_tokens=_dflt_max_new_tokens,eos_token_id:_type_eos_token_id=_dflt_eos_token_id,created_time:_type_created_time=_HAS_DEFAULT_FACTORY,error:_type_error=_dflt_error,lifespan:_type_lifespan=_dflt_lifespan)->_return_type:
  self.request_id=request_id
  self.full_prompt_ids=full_prompt_ids
  self.prompt_ids=prompt_ids
  self.remaining_prompt_ids=_dflt_remaining_prompt_ids() if remaining_prompt_ids is _HAS_DEFAULT_FACTORY else remaining_prompt_ids
  self.static_outputs=_dflt_static_outputs() if static_outputs is _HAS_DEFAULT_FACTORY else static_outputs
  self.allocated_blocks=allocated_blocks
  self.position_offset=position_offset
  self._status=_status
  self.max_new_tokens=max_new_tokens
  self.eos_token_id=eos_token_id
  self.created_time=_dflt_created_time() if created_time is _HAS_DEFAULT_FACTORY else created_time
  self.error=error
  self.lifespan=lifespan
 return __init__