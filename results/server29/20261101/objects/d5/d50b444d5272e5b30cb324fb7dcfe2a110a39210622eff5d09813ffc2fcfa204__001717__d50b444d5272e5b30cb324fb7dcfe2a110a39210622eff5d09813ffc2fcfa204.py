def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.mesh,self.mesh_dim_devices,self.mesh_dim_bandwidth,self.mesh_dim_latency,)==(other.mesh,other.mesh_dim_devices,other.mesh_dim_bandwidth,other.mesh_dim_latency,)
  return NotImplemented
 return __eq__