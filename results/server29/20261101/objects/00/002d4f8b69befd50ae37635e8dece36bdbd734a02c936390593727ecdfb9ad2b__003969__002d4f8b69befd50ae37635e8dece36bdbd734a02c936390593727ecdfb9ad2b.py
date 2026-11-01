def __create_fn__():
 def __repr__(self):
  return self.__class__.__qualname__ + f"(dim={self.dim!r}, broadcast_buffers={self.broadcast_buffers!r}, bucket_cap_mb={self.bucket_cap_mb!r}, find_unused_parameters={self.find_unused_parameters!r}, check_reduction={self.check_reduction!r}, gradient_as_bucket_view={self.gradient_as_bucket_view!r}, static_graph={self.static_graph!r}, comm_hook={self.comm_hook!r}, comm_wrapper={self.comm_wrapper!r}, comm_state_option={self.comm_state_option!r})"
 return __repr__