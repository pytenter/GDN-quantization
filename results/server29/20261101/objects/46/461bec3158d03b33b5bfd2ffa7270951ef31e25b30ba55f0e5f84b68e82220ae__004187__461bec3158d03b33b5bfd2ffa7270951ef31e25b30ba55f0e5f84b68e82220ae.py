def __create_fn__(_type_feature_maps, _type_hidden_states, _type_attentions, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_feature_maps, _dflt_hidden_states, _dflt_attentions, _return_type):
 def __init__(self,feature_maps:_type_feature_maps=_dflt_feature_maps,hidden_states:_type_hidden_states=_dflt_hidden_states,attentions:_type_attentions=_dflt_attentions)->_return_type:
  self.feature_maps=feature_maps
  self.hidden_states=hidden_states
  self.attentions=attentions
  self.__post_init__()
 return __init__