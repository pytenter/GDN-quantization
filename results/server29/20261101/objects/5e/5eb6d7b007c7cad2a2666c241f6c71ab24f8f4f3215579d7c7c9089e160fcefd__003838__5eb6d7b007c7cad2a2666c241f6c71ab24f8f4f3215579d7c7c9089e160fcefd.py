def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.task_type,self.dataset_type,self.dataset_name,self.metric_type,self.metric_value,self.task_name,self.dataset_config,self.dataset_split,self.dataset_revision,self.dataset_args,self.metric_name,self.metric_config,self.metric_args,self.verified,self.verify_token,self.source_name,self.source_url,)==(other.task_type,other.dataset_type,other.dataset_name,other.metric_type,other.metric_value,other.task_name,other.dataset_config,other.dataset_split,other.dataset_revision,other.dataset_args,other.metric_name,other.metric_config,other.metric_args,other.verified,other.verify_token,other.source_name,other.source_url,)
  return NotImplemented
 return __eq__