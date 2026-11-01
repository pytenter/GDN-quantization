def __create_fn__():
 def __repr__(self):
  return self.__class__.__qualname__ + f"(docker_image={self.docker_image!r}, space_id={self.space_id!r}, command={self.command!r}, arguments={self.arguments!r}, environment={self.environment!r}, secrets={self.secrets!r}, flavor={self.flavor!r}, timeout={self.timeout!r}, tags={self.tags!r}, arch={self.arch!r})"
 return __repr__