def __create_fn__(_type_f, _type_args, _type_kwargs, _type_tracked_fakes, _type_name, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_args, _dflt_kwargs, _dflt_tracked_fakes, _dflt_name, _return_type):
 def __init__(self,f:_type_f,args:_type_args=_dflt_args,kwargs:_type_kwargs=_dflt_kwargs,tracked_fakes:_type_tracked_fakes=_dflt_tracked_fakes,name:_type_name=_dflt_name)->_return_type:
  self.f=f
  self.args=args
  self.kwargs=kwargs
  self.tracked_fakes=tracked_fakes
  self.name=name
 return __init__