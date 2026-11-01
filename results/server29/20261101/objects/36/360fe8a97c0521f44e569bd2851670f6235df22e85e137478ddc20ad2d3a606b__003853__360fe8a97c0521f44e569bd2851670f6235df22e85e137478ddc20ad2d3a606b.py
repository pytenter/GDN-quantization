def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.auto_model,self.custom_class,self.pipeline_tag,self.processor,)==(other.auto_model,other.custom_class,other.pipeline_tag,other.processor,)
  return NotImplemented
 return __eq__