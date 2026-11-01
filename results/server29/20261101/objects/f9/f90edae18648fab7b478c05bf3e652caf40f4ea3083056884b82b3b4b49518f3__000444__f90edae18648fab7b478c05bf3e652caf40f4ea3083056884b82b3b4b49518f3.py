def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.originating_source,self.create_fn,self.guard_types,self.code_list,self.obj_weakref,self.guarded_class_weakref,self.stack,self.user_stack,self._hash,)==(other.originating_source,other.create_fn,other.guard_types,other.code_list,other.obj_weakref,other.guarded_class_weakref,other.stack,other.user_stack,other._hash,)
  return NotImplemented
 return __eq__