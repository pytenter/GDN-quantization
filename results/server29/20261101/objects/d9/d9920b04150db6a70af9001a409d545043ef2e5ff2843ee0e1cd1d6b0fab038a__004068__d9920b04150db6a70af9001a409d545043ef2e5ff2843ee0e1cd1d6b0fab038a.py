def __create_fn__():
 def __repr__(self):
  return self.__class__.__qualname__ + f"(sequences={self.sequences!r}, sequences_scores={self.sequences_scores!r}, scores={self.scores!r}, logits={self.logits!r}, beam_indices={self.beam_indices!r}, encoder_attentions={self.encoder_attentions!r}, encoder_hidden_states={self.encoder_hidden_states!r}, decoder_attentions={self.decoder_attentions!r}, cross_attentions={self.cross_attentions!r}, decoder_hidden_states={self.decoder_hidden_states!r}, past_key_values={self.past_key_values!r})"
 return __repr__