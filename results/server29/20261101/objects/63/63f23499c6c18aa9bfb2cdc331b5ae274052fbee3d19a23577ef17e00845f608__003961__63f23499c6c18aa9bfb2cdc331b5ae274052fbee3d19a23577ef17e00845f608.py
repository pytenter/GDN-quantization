def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.input_ids,self.attention_mask,self.position_ids,self.cumulative_seqlens_q,self.cumulative_seqlens_k,self.max_seqlen_q,self.max_seqlen_k,self.write_index,self.read_index,self.logits_indices,self.cache,self.use_cache,)==(other.input_ids,other.attention_mask,other.position_ids,other.cumulative_seqlens_q,other.cumulative_seqlens_k,other.max_seqlen_q,other.max_seqlen_k,other.write_index,other.read_index,other.logits_indices,other.cache,other.use_cache,)
  return NotImplemented
 return __eq__