def __create_fn__(_type_init_scale, _type_growth_factor, _type_backoff_factor, _type_growth_interval, _type_enabled, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_init_scale, _dflt_growth_factor, _dflt_backoff_factor, _dflt_growth_interval, _dflt_enabled, _return_type):
 def __init__(self,init_scale:_type_init_scale=_dflt_init_scale,growth_factor:_type_growth_factor=_dflt_growth_factor,backoff_factor:_type_backoff_factor=_dflt_backoff_factor,growth_interval:_type_growth_interval=_dflt_growth_interval,enabled:_type_enabled=_dflt_enabled)->_return_type:
  self.init_scale=init_scale
  self.growth_factor=growth_factor
  self.backoff_factor=backoff_factor
  self.growth_interval=growth_interval
  self.enabled=enabled
 return __init__