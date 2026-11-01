def __create_fn__(_type_index, _type_size, _type_pred_buffers, _type_pred_nodes, _type_succ_nodes, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_index, _dflt_size, _dflt_pred_buffers, _dflt_pred_nodes, _dflt_succ_nodes, _return_type):
 def __init__(self,index:_type_index=_dflt_index,size:_type_size=_dflt_size,pred_buffers:_type_pred_buffers=_HAS_DEFAULT_FACTORY,pred_nodes:_type_pred_nodes=_HAS_DEFAULT_FACTORY,succ_nodes:_type_succ_nodes=_HAS_DEFAULT_FACTORY)->_return_type:
  self.index=index
  self.size=size
  self.pred_buffers=_dflt_pred_buffers() if pred_buffers is _HAS_DEFAULT_FACTORY else pred_buffers
  self.pred_nodes=_dflt_pred_nodes() if pred_nodes is _HAS_DEFAULT_FACTORY else pred_nodes
  self.succ_nodes=_dflt_succ_nodes() if succ_nodes is _HAS_DEFAULT_FACTORY else succ_nodes
 return __init__