def __create_fn__(_type_docker_image, _type_space_id, _type_command, _type_arguments, _type_environment, _type_secrets, _type_flavor, _type_timeout, _type_tags, _type_arch, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _return_type):
 def __init__(self,docker_image:_type_docker_image,space_id:_type_space_id,command:_type_command,arguments:_type_arguments,environment:_type_environment,secrets:_type_secrets,flavor:_type_flavor,timeout:_type_timeout,tags:_type_tags,arch:_type_arch)->_return_type:
  self.docker_image=docker_image
  self.space_id=space_id
  self.command=command
  self.arguments=arguments
  self.environment=environment
  self.secrets=secrets
  self.flavor=flavor
  self.timeout=timeout
  self.tags=tags
  self.arch=arch
 return __init__