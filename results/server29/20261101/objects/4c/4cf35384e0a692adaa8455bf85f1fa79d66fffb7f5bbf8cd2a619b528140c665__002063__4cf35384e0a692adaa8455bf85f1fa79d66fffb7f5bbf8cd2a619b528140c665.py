def __create_fn__(_type_repository_uri, _type_as_of_time_utc, _type_branch, _type_mapped_to, _type_properties, _type_revision_id, _type_revision_tag, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_as_of_time_utc, _dflt_branch, _dflt_mapped_to, _dflt_properties, _dflt_revision_id, _dflt_revision_tag, _return_type):
 def __init__(self,repository_uri:_type_repository_uri,as_of_time_utc:_type_as_of_time_utc=_dflt_as_of_time_utc,branch:_type_branch=_dflt_branch,mapped_to:_type_mapped_to=_dflt_mapped_to,properties:_type_properties=_dflt_properties,revision_id:_type_revision_id=_dflt_revision_id,revision_tag:_type_revision_tag=_dflt_revision_tag)->_return_type:
  self.repository_uri=repository_uri
  self.as_of_time_utc=as_of_time_utc
  self.branch=branch
  self.mapped_to=mapped_to
  self.properties=properties
  self.revision_id=revision_id
  self.revision_tag=revision_tag
 return __init__