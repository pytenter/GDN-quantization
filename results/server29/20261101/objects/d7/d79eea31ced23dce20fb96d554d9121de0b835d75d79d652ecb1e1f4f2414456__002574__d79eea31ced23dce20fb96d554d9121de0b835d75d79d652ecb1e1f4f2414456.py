def __create_fn__(_type_fusible_ops, _type_compute_intensive_ops, _type_random_ops, _type_view_ops, _type_recomputable_ops, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _return_type):
 def __init__(self,fusible_ops:_type_fusible_ops,compute_intensive_ops:_type_compute_intensive_ops,random_ops:_type_random_ops,view_ops:_type_view_ops,recomputable_ops:_type_recomputable_ops)->_return_type:
  self.fusible_ops=fusible_ops
  self.compute_intensive_ops=compute_intensive_ops
  self.random_ops=random_ops
  self.view_ops=view_ops
  self.recomputable_ops=recomputable_ops
 return __init__