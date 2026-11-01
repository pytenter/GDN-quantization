def __create_fn__(_type_auto_model, _type_custom_class, _type_pipeline_tag, _type_processor, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_custom_class, _dflt_pipeline_tag, _dflt_processor, _return_type):
 def __init__(self,auto_model:_type_auto_model,custom_class:_type_custom_class=_dflt_custom_class,pipeline_tag:_type_pipeline_tag=_dflt_pipeline_tag,processor:_type_processor=_dflt_processor)->_return_type:
  self.auto_model=auto_model
  self.custom_class=custom_class
  self.pipeline_tag=pipeline_tag
  self.processor=processor
  self.__post_init__()
 return __init__