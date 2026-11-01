def __create_fn__():
 def __repr__(self):
  return self.__class__.__qualname__ + f"(id={self.id!r}, created_at={self.created_at!r}, job_spec={self.job_spec!r}, schedule={self.schedule!r}, suspend={self.suspend!r}, concurrency={self.concurrency!r}, status={self.status!r}, owner={self.owner!r})"
 return __repr__