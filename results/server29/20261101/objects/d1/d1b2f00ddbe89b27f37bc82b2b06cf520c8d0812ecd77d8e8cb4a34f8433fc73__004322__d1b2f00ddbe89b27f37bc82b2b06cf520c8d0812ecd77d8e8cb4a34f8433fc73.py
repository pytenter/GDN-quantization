def __create_fn__(_type_loss, _type_logits, _type_past_key_values, _type_hidden_states, _type_attentions, _type_z_loss, _type_aux_loss, _type_router_logits, _type_mtp_loss, _type_mtp_logits, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_loss, _dflt_logits, _dflt_past_key_values, _dflt_hidden_states, _dflt_attentions, _dflt_z_loss, _dflt_aux_loss, _dflt_router_logits, _dflt_mtp_loss, _dflt_mtp_logits, _return_type):
 def __init__(self,loss:_type_loss=_dflt_loss,logits:_type_logits=_dflt_logits,past_key_values:_type_past_key_values=_dflt_past_key_values,hidden_states:_type_hidden_states=_dflt_hidden_states,attentions:_type_attentions=_dflt_attentions,z_loss:_type_z_loss=_dflt_z_loss,aux_loss:_type_aux_loss=_dflt_aux_loss,router_logits:_type_router_logits=_dflt_router_logits,mtp_loss:_type_mtp_loss=_dflt_mtp_loss,mtp_logits:_type_mtp_logits=_dflt_mtp_logits)->_return_type:
  self.loss=loss
  self.logits=logits
  self.past_key_values=past_key_values
  self.hidden_states=hidden_states
  self.attentions=attentions
  self.z_loss=z_loss
  self.aux_loss=aux_loss
  self.router_logits=router_logits
  self.mtp_loss=mtp_loss
  self.mtp_logits=mtp_logits
  self.__post_init__()
 return __init__