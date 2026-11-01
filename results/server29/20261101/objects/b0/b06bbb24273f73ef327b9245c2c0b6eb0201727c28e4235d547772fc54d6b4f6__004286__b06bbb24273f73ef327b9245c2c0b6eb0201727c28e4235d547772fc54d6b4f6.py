def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.quant_method,self.quant_type,self.modules_to_not_convert,self.quant_type_kwargs,self.include_input_output_embeddings,self.untie_embedding_weights,)==(other.quant_method,other.quant_type,other.modules_to_not_convert,other.quant_type_kwargs,other.include_input_output_embeddings,other.untie_embedding_weights,)
  return NotImplemented
 return __eq__