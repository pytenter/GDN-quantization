def __create_fn__():
 def __repr__(self):
  return self.__class__.__qualname__ + f"(id={self.id!r}, created_at={self.created_at!r}, docker_image={self.docker_image!r}, space_id={self.space_id!r}, command={self.command!r}, arguments={self.arguments!r}, environment={self.environment!r}, secrets={self.secrets!r}, flavor={self.flavor!r}, status={self.status!r}, owner={self.owner!r}, endpoint={self.endpoint!r}, url={self.url!r})"
 return __repr__