def __create_fn__(_type_name, _type_layout, _type_inputs, _type_constant_args, _type_kwargs, _type_output_view, _type_python_kernel_name, _type_cpp_kernel_name, _type_ordered_kwargs_for_cpp_kernel, _type_op_overload, _type_arg_properties, _type_kwarg_properties, _type_unbacked_bindings, _type_mutation_outputs, _type_subgraph, _type_operands, _type_outputs, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_constant_args, _dflt_kwargs, _dflt_output_view, _dflt_python_kernel_name, _dflt_cpp_kernel_name, _dflt_ordered_kwargs_for_cpp_kernel, _dflt_op_overload, _dflt_arg_properties, _dflt_kwarg_properties, _dflt_unbacked_bindings, _dflt_mutation_outputs, _dflt_subgraph, _dflt_operands, _dflt_outputs, _return_type):
 def __init__(self,*,name:_type_name,layout:_type_layout,inputs:_type_inputs,constant_args:_type_constant_args=_dflt_constant_args,kwargs:_type_kwargs=_HAS_DEFAULT_FACTORY,output_view:_type_output_view=_dflt_output_view,python_kernel_name:_type_python_kernel_name=_dflt_python_kernel_name,cpp_kernel_name:_type_cpp_kernel_name=_dflt_cpp_kernel_name,ordered_kwargs_for_cpp_kernel:_type_ordered_kwargs_for_cpp_kernel=_HAS_DEFAULT_FACTORY,op_overload:_type_op_overload=_dflt_op_overload,arg_properties:_type_arg_properties=_dflt_arg_properties,kwarg_properties:_type_kwarg_properties=_dflt_kwarg_properties,unbacked_bindings:_type_unbacked_bindings=_HAS_DEFAULT_FACTORY,mutation_outputs:_type_mutation_outputs=_HAS_DEFAULT_FACTORY,subgraph:_type_subgraph=_dflt_subgraph,operands:_type_operands=_dflt_operands,outputs:_type_outputs=_dflt_outputs)->_return_type:
  self.name=name
  self.layout=layout
  self.inputs=inputs
  self.constant_args=constant_args
  self.kwargs=_dflt_kwargs() if kwargs is _HAS_DEFAULT_FACTORY else kwargs
  self.output_view=output_view
  self.python_kernel_name=python_kernel_name
  self.cpp_kernel_name=cpp_kernel_name
  self.ordered_kwargs_for_cpp_kernel=_dflt_ordered_kwargs_for_cpp_kernel() if ordered_kwargs_for_cpp_kernel is _HAS_DEFAULT_FACTORY else ordered_kwargs_for_cpp_kernel
  self.op_overload=op_overload
  self.arg_properties=arg_properties
  self.kwarg_properties=kwarg_properties
  self.unbacked_bindings=_dflt_unbacked_bindings() if unbacked_bindings is _HAS_DEFAULT_FACTORY else unbacked_bindings
  self.mutation_outputs=_dflt_mutation_outputs() if mutation_outputs is _HAS_DEFAULT_FACTORY else mutation_outputs
  self.subgraph=subgraph
  self.operands=operands
  self.outputs=outputs
  self.__post_init__()
 return __init__