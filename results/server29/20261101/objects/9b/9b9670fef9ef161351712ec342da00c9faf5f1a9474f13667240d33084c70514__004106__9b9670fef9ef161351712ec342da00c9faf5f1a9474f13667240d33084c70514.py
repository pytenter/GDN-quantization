def __create_fn__(_type_last_hidden_state, _type_hidden_states, _type_attentions, _type_router_probs, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_last_hidden_state, _dflt_hidden_states, _dflt_attentions, _dflt_router_probs, _return_type):
 def __init__(self,last_hidden_state:_type_last_hidden_state=_dflt_last_hidden_state,hidden_states:_type_hidden_states=_dflt_hidden_states,attentions:_type_attentions=_dflt_attentions,router_probs:_type_router_probs=_dflt_router_probs)->_return_type:
  self.last_hidden_state=last_hidden_state
  self.hidden_states=hidden_states
  self.attentions=attentions
  self.router_probs=router_probs
  self.__post_init__()
 return __init__