def __create_fn__():
 def __repr__(self):
  return self.__class__.__qualname__ + f"(loss={self.loss!r}, logits={self.logits!r}, past_key_values={self.past_key_values!r}, hidden_states={self.hidden_states!r}, attentions={self.attentions!r}, z_loss={self.z_loss!r}, aux_loss={self.aux_loss!r}, router_logits={self.router_logits!r}, mtp_loss={self.mtp_loss!r}, mtp_logits={self.mtp_logits!r})"
 return __repr__