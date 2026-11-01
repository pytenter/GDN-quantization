def __create_fn__(_type_task_type, _type_dataset_type, _type_dataset_name, _type_metric_type, _type_metric_value, _type_task_name, _type_dataset_config, _type_dataset_split, _type_dataset_revision, _type_dataset_args, _type_metric_name, _type_metric_config, _type_metric_args, _type_verified, _type_verify_token, _type_source_name, _type_source_url, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_task_name, _dflt_dataset_config, _dflt_dataset_split, _dflt_dataset_revision, _dflt_dataset_args, _dflt_metric_name, _dflt_metric_config, _dflt_metric_args, _dflt_verified, _dflt_verify_token, _dflt_source_name, _dflt_source_url, _return_type):
 def __init__(self,task_type:_type_task_type,dataset_type:_type_dataset_type,dataset_name:_type_dataset_name,metric_type:_type_metric_type,metric_value:_type_metric_value,task_name:_type_task_name=_dflt_task_name,dataset_config:_type_dataset_config=_dflt_dataset_config,dataset_split:_type_dataset_split=_dflt_dataset_split,dataset_revision:_type_dataset_revision=_dflt_dataset_revision,dataset_args:_type_dataset_args=_dflt_dataset_args,metric_name:_type_metric_name=_dflt_metric_name,metric_config:_type_metric_config=_dflt_metric_config,metric_args:_type_metric_args=_dflt_metric_args,verified:_type_verified=_dflt_verified,verify_token:_type_verify_token=_dflt_verify_token,source_name:_type_source_name=_dflt_source_name,source_url:_type_source_url=_dflt_source_url)->_return_type:
  self.task_type=task_type
  self.dataset_type=dataset_type
  self.dataset_name=dataset_name
  self.metric_type=metric_type
  self.metric_value=metric_value
  self.task_name=task_name
  self.dataset_config=dataset_config
  self.dataset_split=dataset_split
  self.dataset_revision=dataset_revision
  self.dataset_args=dataset_args
  self.metric_name=metric_name
  self.metric_config=metric_config
  self.metric_args=metric_args
  self.verified=verified
  self.verify_token=verify_token
  self.source_name=source_name
  self.source_url=source_url
  self.__post_init__()
 return __init__