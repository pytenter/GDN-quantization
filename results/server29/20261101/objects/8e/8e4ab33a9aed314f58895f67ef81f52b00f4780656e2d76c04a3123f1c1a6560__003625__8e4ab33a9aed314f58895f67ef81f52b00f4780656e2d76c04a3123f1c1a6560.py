def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.ihook,self.msg,self.kwargs,self.suspend_capture_ctx,self.request,self.config,)==(other.ihook,other.msg,other.kwargs,other.suspend_capture_ctx,other.request,other.config,)
  return NotImplemented
 return __eq__