def __create_fn__(cls, FrozenInstanceError):
 def __delattr__(self,name):
  if type(self) is cls or name in ('derived_from','derive_qparams_fn','dtype','quant_min','quant_max','qscheme','ch_axis','is_dynamic',):
   raise FrozenInstanceError(f"cannot delete field {name!r}")
  super(cls, self).__delattr__(name)
 return __delattr__