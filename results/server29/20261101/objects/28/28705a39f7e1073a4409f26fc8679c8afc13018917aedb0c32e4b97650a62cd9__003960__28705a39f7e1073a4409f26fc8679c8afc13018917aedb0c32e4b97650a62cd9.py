def __create_fn__():
 def __repr__(self):
  return self.__class__.__qualname__ + f"(input_ids={self.input_ids!r}, attention_mask={self.attention_mask!r}, position_ids={self.position_ids!r}, cumulative_seqlens_q={self.cumulative_seqlens_q!r}, cumulative_seqlens_k={self.cumulative_seqlens_k!r}, max_seqlen_q={self.max_seqlen_q!r}, max_seqlen_k={self.max_seqlen_k!r}, write_index={self.write_index!r}, read_index={self.read_index!r}, logits_indices={self.logits_indices!r}, cache={self.cache!r}, use_cache={self.use_cache!r})"
 return __repr__