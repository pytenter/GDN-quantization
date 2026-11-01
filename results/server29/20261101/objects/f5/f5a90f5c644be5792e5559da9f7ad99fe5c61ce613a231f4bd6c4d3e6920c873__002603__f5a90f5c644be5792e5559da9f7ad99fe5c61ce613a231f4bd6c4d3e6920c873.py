def __create_fn__(_type_name, _type_config_name, _type_config_field, _type_config_value, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _return_type):
 def __init__(self,config_name:_type_config_name,config_field:_type_config_field,config_value:_type_config_value)->_return_type:
  self.config_name=config_name
  self.config_field=config_field
  self.config_value=config_value
  self.__post_init__()
 return __init__