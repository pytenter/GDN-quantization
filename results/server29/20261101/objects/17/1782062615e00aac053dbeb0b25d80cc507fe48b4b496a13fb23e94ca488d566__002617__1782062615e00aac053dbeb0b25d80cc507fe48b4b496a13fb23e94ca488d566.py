def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.graphs,self.graph_count,self.graph_break_count,self.break_reasons,self.op_count,self.ops_per_graph,self.out_guards,self.compile_times,)==(other.graphs,other.graph_count,other.graph_break_count,other.break_reasons,other.op_count,other.ops_per_graph,other.out_guards,other.compile_times,)
  return NotImplemented
 return __eq__