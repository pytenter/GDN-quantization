def __create_fn__():
 def __hash__(self):
  return hash((self.access_token,self.expiration_unix_epoch,self.endpoint,))
 return __hash__