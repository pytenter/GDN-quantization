def __create_fn__(_type_sequences, _type_sequences_scores, _type_scores, _type_logits, _type_beam_indices, _type_attentions, _type_hidden_states, _type_past_key_values, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_sequences_scores, _dflt_scores, _dflt_logits, _dflt_beam_indices, _dflt_attentions, _dflt_hidden_states, _dflt_past_key_values, _return_type):
 def __init__(self,sequences:_type_sequences,sequences_scores:_type_sequences_scores=_dflt_sequences_scores,scores:_type_scores=_dflt_scores,logits:_type_logits=_dflt_logits,beam_indices:_type_beam_indices=_dflt_beam_indices,attentions:_type_attentions=_dflt_attentions,hidden_states:_type_hidden_states=_dflt_hidden_states,past_key_values:_type_past_key_values=_dflt_past_key_values)->_return_type:
  self.sequences=sequences
  self.sequences_scores=sequences_scores
  self.scores=scores
  self.logits=logits
  self.beam_indices=beam_indices
  self.attentions=attentions
  self.hidden_states=hidden_states
  self.past_key_values=past_key_values
  self.__post_init__()
 return __init__