def __create_fn__(_type_loss, _type_logits, _type_hidden_states, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_loss, _dflt_logits, _dflt_hidden_states, _return_type):
 def __init__(self,loss:_type_loss=_dflt_loss,logits:_type_logits=_dflt_logits,hidden_states:_type_hidden_states=_dflt_hidden_states)->_return_type:
  self.loss=loss
  self.logits=logits
  self.hidden_states=hidden_states
  self.__post_init__()
 return __init__