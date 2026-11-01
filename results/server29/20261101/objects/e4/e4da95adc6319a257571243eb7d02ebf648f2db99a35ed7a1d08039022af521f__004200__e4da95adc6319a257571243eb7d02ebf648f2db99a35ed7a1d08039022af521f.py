def __create_fn__():
 def __repr__(self):
  return self.__class__.__qualname__ + f"(loss={self.loss!r}, params={self.params!r}, past_key_values={self.past_key_values!r}, decoder_hidden_states={self.decoder_hidden_states!r}, decoder_attentions={self.decoder_attentions!r}, cross_attentions={self.cross_attentions!r}, encoder_last_hidden_state={self.encoder_last_hidden_state!r}, encoder_hidden_states={self.encoder_hidden_states!r}, encoder_attentions={self.encoder_attentions!r}, loc={self.loc!r}, scale={self.scale!r}, static_features={self.static_features!r})"
 return __repr__