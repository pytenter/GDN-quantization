def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.name,self.layout,self.inputs,self.constant_args,self.kwargs,self.output_view,self.python_kernel_name,self.cpp_kernel_name,self.ordered_kwargs_for_cpp_kernel,self.op_overload,self.arg_properties,self.kwarg_properties,self.unbacked_bindings,self.mutation_outputs,self.predicate,self.operands,self.true_subgraph,self.false_subgraph,self.outputs,)==(other.name,other.layout,other.inputs,other.constant_args,other.kwargs,other.output_view,other.python_kernel_name,other.cpp_kernel_name,other.ordered_kwargs_for_cpp_kernel,other.op_overload,other.arg_properties,other.kwarg_properties,other.unbacked_bindings,other.mutation_outputs,other.predicate,other.operands,other.true_subgraph,other.false_subgraph,other.outputs,)
  return NotImplemented
 return __eq__