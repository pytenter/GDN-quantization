def __create_fn__(_type_loss, _type_predicted_depth, _type_hidden_states, _type_attentions, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_loss, _dflt_predicted_depth, _dflt_hidden_states, _dflt_attentions, _return_type):
 def __init__(self,loss:_type_loss=_dflt_loss,predicted_depth:_type_predicted_depth=_dflt_predicted_depth,hidden_states:_type_hidden_states=_dflt_hidden_states,attentions:_type_attentions=_dflt_attentions)->_return_type:
  self.loss=loss
  self.predicted_depth=predicted_depth
  self.hidden_states=hidden_states
  self.attentions=attentions
  self.__post_init__()
 return __init__