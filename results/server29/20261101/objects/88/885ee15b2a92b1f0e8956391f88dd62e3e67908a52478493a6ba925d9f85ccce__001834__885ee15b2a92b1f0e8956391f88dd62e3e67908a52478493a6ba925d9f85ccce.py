def __create_fn__(_type_module, _type_param_name, _type_shared_modules, _type_shared_param_names, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_shared_modules, _dflt_shared_param_names, _return_type):
 def __init__(self,module:_type_module,param_name:_type_param_name,shared_modules:_type_shared_modules=_HAS_DEFAULT_FACTORY,shared_param_names:_type_shared_param_names=_HAS_DEFAULT_FACTORY)->_return_type:
  self.module=module
  self.param_name=param_name
  self.shared_modules=_dflt_shared_modules() if shared_modules is _HAS_DEFAULT_FACTORY else shared_modules
  self.shared_param_names=_dflt_shared_param_names() if shared_param_names is _HAS_DEFAULT_FACTORY else shared_param_names
 return __init__