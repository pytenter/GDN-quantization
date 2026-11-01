def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.provider,self.hf_model_id,self.provider_id,self.status,self.task,self.adapter,self.adapter_weights_path,self.type,)==(other.provider,other.hf_model_id,other.provider_id,other.status,other.task,other.adapter,other.adapter_weights_path,other.type,)
  return NotImplemented
 return __eq__