def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.fd,self.creator_pid,self.device,self.inode,)==(other.fd,other.creator_pid,other.device,other.inode,)
  return NotImplemented
 return __eq__