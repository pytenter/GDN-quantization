def __create_fn__():
 def __repr__(self):
  return self.__class__.__qualname__ + f"(rule={self.rule!r}, level={self.level!r}, message={self.message!r}, locations={self.locations!r}, stacks={self.stacks!r}, graphs={self.graphs!r}, thread_flow_locations={self.thread_flow_locations!r}, additional_messages={self.additional_messages!r}, tags={self.tags!r}, source_exception={self.source_exception!r}, logger={self.logger!r}, _current_log_section_depth={self._current_log_section_depth!r})"
 return __repr__