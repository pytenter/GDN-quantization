def __create_fn__(_type_last_hidden_state, _type_pooler_output, _type_hidden_states, _type_attentions, _type_projection_state, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_last_hidden_state, _dflt_pooler_output, _dflt_hidden_states, _dflt_attentions, _dflt_projection_state, _return_type):
 def __init__(self,last_hidden_state:_type_last_hidden_state=_dflt_last_hidden_state,pooler_output:_type_pooler_output=_dflt_pooler_output,hidden_states:_type_hidden_states=_dflt_hidden_states,attentions:_type_attentions=_dflt_attentions,projection_state:_type_projection_state=_dflt_projection_state)->_return_type:
  self.last_hidden_state=last_hidden_state
  self.pooler_output=pooler_output
  self.hidden_states=hidden_states
  self.attentions=attentions
  self.projection_state=projection_state
  self.__post_init__()
 return __init__