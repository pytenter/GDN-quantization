def __create_fn__(_type_source, _type__example, _type_pass_arg_as_tensor, _type_fake_tensor, _type_is_tensor, _type_example_strong_ref, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_is_tensor, _dflt_example_strong_ref, _return_type):
 def __init__(self,source:_type_source,_example:_type__example,pass_arg_as_tensor:_type_pass_arg_as_tensor,fake_tensor:_type_fake_tensor,is_tensor:_type_is_tensor=_dflt_is_tensor,example_strong_ref:_type_example_strong_ref=_dflt_example_strong_ref)->_return_type:
  self.source=source
  self._example=_example
  self.pass_arg_as_tensor=pass_arg_as_tensor
  self.fake_tensor=fake_tensor
  self.is_tensor=is_tensor
  self.example_strong_ref=example_strong_ref
  self.__post_init__()
 return __init__