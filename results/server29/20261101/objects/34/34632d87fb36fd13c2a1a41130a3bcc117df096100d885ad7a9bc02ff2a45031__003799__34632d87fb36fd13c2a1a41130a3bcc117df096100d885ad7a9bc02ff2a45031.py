def __create_fn__(_type_id, _type_created_at, _type_docker_image, _type_space_id, _type_command, _type_arguments, _type_environment, _type_secrets, _type_flavor, _type_status, _type_owner, _type_endpoint, _type_url, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _return_type):
 def __init__(self,id:_type_id,created_at:_type_created_at,docker_image:_type_docker_image,space_id:_type_space_id,command:_type_command,arguments:_type_arguments,environment:_type_environment,secrets:_type_secrets,flavor:_type_flavor,status:_type_status,owner:_type_owner,endpoint:_type_endpoint,url:_type_url)->_return_type:
  self.id=id
  self.created_at=created_at
  self.docker_image=docker_image
  self.space_id=space_id
  self.command=command
  self.arguments=arguments
  self.environment=environment
  self.secrets=secrets
  self.flavor=flavor
  self.status=status
  self.owner=owner
  self.endpoint=endpoint
  self.url=url
 return __init__