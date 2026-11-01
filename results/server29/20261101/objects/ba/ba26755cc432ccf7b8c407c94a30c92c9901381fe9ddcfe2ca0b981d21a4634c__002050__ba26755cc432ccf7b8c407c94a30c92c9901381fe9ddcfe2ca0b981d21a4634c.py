def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.addresses,self.artifacts,self.conversion,self.driver,self.extensions,self.externalized_properties,self.graphs,self.guid,self.invocations,self.logical_locations,self.policies,self.properties,self.results,self.run_guid,self.schema,self.taxonomies,self.thread_flow_locations,self.translations,self.version,self.web_requests,self.web_responses,)==(other.addresses,other.artifacts,other.conversion,other.driver,other.extensions,other.externalized_properties,other.graphs,other.guid,other.invocations,other.logical_locations,other.policies,other.properties,other.results,other.run_guid,other.schema,other.taxonomies,other.thread_flow_locations,other.translations,other.version,other.web_requests,other.web_responses,)
  return NotImplemented
 return __eq__