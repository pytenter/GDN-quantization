def __create_fn__(_type_originating_source, _type_create_fn, _type_guard_types, _type_code_list, _type_obj_weakref, _type_guarded_class_weakref, _type_stack, _type_user_stack, _type__hash, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_guard_types, _dflt_code_list, _dflt_obj_weakref, _dflt_guarded_class_weakref, _dflt_stack, _dflt_user_stack, _dflt__hash, _return_type):
 def __init__(self,originating_source:_type_originating_source,create_fn:_type_create_fn,guard_types:_type_guard_types=_dflt_guard_types,code_list:_type_code_list=_dflt_code_list,obj_weakref:_type_obj_weakref=_dflt_obj_weakref,guarded_class_weakref:_type_guarded_class_weakref=_dflt_guarded_class_weakref,stack:_type_stack=_dflt_stack,user_stack:_type_user_stack=_dflt_user_stack,_hash:_type__hash=_dflt__hash)->_return_type:
  self.originating_source=originating_source
  self.create_fn=create_fn
  self.guard_types=guard_types
  self.code_list=code_list
  self.obj_weakref=obj_weakref
  self.guarded_class_weakref=guarded_class_weakref
  self.stack=stack
  self.user_stack=user_stack
  self._hash=_hash
 return __init__