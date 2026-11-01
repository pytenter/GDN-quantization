def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.log_alias_to_log_qnames,self.artifact_log_qnames,self.child_log_qnames,self.artifact_names,self.visible_artifacts,self.artifact_descriptions,self.off_by_default_artifact_names,self.artifact_log_formatters,)==(other.log_alias_to_log_qnames,other.artifact_log_qnames,other.child_log_qnames,other.artifact_names,other.visible_artifacts,other.artifact_descriptions,other.off_by_default_artifact_names,other.artifact_log_formatters,)
  return NotImplemented
 return __eq__