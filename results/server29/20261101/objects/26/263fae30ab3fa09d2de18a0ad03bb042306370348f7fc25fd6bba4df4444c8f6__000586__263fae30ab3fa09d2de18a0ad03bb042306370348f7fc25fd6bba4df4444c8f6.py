def __create_fn__(_type_graph, _type_order, _type_name, _type_input_placeholders, _type_orig_inputs, _type_orig_outputs, _type_getattr_maps, _type_constructor_args, _type_gm, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_input_placeholders, _dflt_orig_inputs, _dflt_orig_outputs, _dflt_getattr_maps, _dflt_constructor_args, _dflt_gm, _return_type):
 def __init__(self,graph:_type_graph,order:_type_order,name:_type_name,input_placeholders:_type_input_placeholders=_HAS_DEFAULT_FACTORY,orig_inputs:_type_orig_inputs=_HAS_DEFAULT_FACTORY,orig_outputs:_type_orig_outputs=_HAS_DEFAULT_FACTORY,getattr_maps:_type_getattr_maps=_HAS_DEFAULT_FACTORY,constructor_args:_type_constructor_args=_HAS_DEFAULT_FACTORY,gm:_type_gm=_dflt_gm)->_return_type:
  self.graph=graph
  self.order=order
  self.name=name
  self.input_placeholders=_dflt_input_placeholders() if input_placeholders is _HAS_DEFAULT_FACTORY else input_placeholders
  self.orig_inputs=_dflt_orig_inputs() if orig_inputs is _HAS_DEFAULT_FACTORY else orig_inputs
  self.orig_outputs=_dflt_orig_outputs() if orig_outputs is _HAS_DEFAULT_FACTORY else orig_outputs
  self.getattr_maps=_dflt_getattr_maps() if getattr_maps is _HAS_DEFAULT_FACTORY else getattr_maps
  self.constructor_args=_dflt_constructor_args() if constructor_args is _HAS_DEFAULT_FACTORY else constructor_args
  self.gm=gm
 return __init__