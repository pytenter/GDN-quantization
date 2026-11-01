def __create_fn__(_type_input_ids, _type_attention_mask, _type_position_ids, _type_cumulative_seqlens_q, _type_cumulative_seqlens_k, _type_max_seqlen_q, _type_max_seqlen_k, _type_write_index, _type_read_index, _type_logits_indices, _type_cache, _type_use_cache, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_use_cache, _return_type):
 def __init__(self,input_ids:_type_input_ids,attention_mask:_type_attention_mask,position_ids:_type_position_ids,cumulative_seqlens_q:_type_cumulative_seqlens_q,cumulative_seqlens_k:_type_cumulative_seqlens_k,max_seqlen_q:_type_max_seqlen_q,max_seqlen_k:_type_max_seqlen_k,write_index:_type_write_index,read_index:_type_read_index,logits_indices:_type_logits_indices,cache:_type_cache,use_cache:_type_use_cache=_dflt_use_cache)->_return_type:
  self.input_ids=input_ids
  self.attention_mask=attention_mask
  self.position_ids=position_ids
  self.cumulative_seqlens_q=cumulative_seqlens_q
  self.cumulative_seqlens_k=cumulative_seqlens_k
  self.max_seqlen_q=max_seqlen_q
  self.max_seqlen_k=max_seqlen_k
  self.write_index=write_index
  self.read_index=read_index
  self.logits_indices=logits_indices
  self.cache=cache
  self.use_cache=use_cache
 return __init__