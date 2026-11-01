def __create_fn__(_type_provider, _type_hf_model_id, _type_provider_id, _type_status, _type_task, _type_adapter, _type_adapter_weights_path, _type_type, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_adapter, _dflt_adapter_weights_path, _dflt_type, _return_type):
 def __init__(self,provider:_type_provider,hf_model_id:_type_hf_model_id,provider_id:_type_provider_id,status:_type_status,task:_type_task,adapter:_type_adapter=_dflt_adapter,adapter_weights_path:_type_adapter_weights_path=_dflt_adapter_weights_path,type:_type_type=_dflt_type)->_return_type:
  self.provider=provider
  self.hf_model_id=hf_model_id
  self.provider_id=provider_id
  self.status=status
  self.task=task
  self.adapter=adapter
  self.adapter_weights_path=adapter_weights_path
  self.type=type
 return __init__