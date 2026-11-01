def __create_fn__(_type_loss, _type_start_logits, _type_end_logits, _type_hidden_states, _type_attentions, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_loss, _dflt_start_logits, _dflt_end_logits, _dflt_hidden_states, _dflt_attentions, _return_type):
 def __init__(self,loss:_type_loss=_dflt_loss,start_logits:_type_start_logits=_dflt_start_logits,end_logits:_type_end_logits=_dflt_end_logits,hidden_states:_type_hidden_states=_dflt_hidden_states,attentions:_type_attentions=_dflt_attentions)->_return_type:
  self.loss=loss
  self.start_logits=start_logits
  self.end_logits=end_logits
  self.hidden_states=hidden_states
  self.attentions=attentions
  self.__post_init__()
 return __init__