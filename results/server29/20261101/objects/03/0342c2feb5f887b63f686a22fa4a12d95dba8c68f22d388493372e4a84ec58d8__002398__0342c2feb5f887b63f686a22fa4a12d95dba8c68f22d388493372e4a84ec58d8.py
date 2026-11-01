def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.lock_file,self.timeout,self.mode,self.blocking,self.poll_interval,self.lifetime,self.lock_file_fd,self.lock_file_fd_token,self.lock_file_fd_identity,self.pending_lock_file_fd,self.pending_lock_file_fd_identity,self.lock_counter,self.lock_file_key,self.owner_claim_paths,self.claim_root,self.run_in_executor,self.executor,self.loop,)==(other.lock_file,other.timeout,other.mode,other.blocking,other.poll_interval,other.lifetime,other.lock_file_fd,other.lock_file_fd_token,other.lock_file_fd_identity,other.pending_lock_file_fd,other.pending_lock_file_fd_identity,other.lock_counter,other.lock_file_key,other.owner_claim_paths,other.claim_root,other.run_in_executor,other.executor,other.loop,)
  return NotImplemented
 return __eq__