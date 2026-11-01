def __create_fn__(_type_graphs, _type_graph_count, _type_graph_break_count, _type_break_reasons, _type_op_count, _type_ops_per_graph, _type_out_guards, _type_compile_times, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_ops_per_graph, _dflt_out_guards, _dflt_compile_times, _return_type):
 def __init__(self,graphs:_type_graphs,graph_count:_type_graph_count,graph_break_count:_type_graph_break_count,break_reasons:_type_break_reasons,op_count:_type_op_count,ops_per_graph:_type_ops_per_graph=_dflt_ops_per_graph,out_guards:_type_out_guards=_dflt_out_guards,compile_times:_type_compile_times=_dflt_compile_times)->_return_type:
  self.graphs=graphs
  self.graph_count=graph_count
  self.graph_break_count=graph_break_count
  self.break_reasons=break_reasons
  self.op_count=op_count
  self.ops_per_graph=ops_per_graph
  self.out_guards=out_guards
  self.compile_times=compile_times
 return __init__