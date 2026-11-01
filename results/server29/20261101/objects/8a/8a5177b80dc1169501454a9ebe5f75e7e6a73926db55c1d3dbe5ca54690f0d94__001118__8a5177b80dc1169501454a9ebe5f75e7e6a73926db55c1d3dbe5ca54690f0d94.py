def __create_fn__(cls, FrozenInstanceError):
 def __setattr__(self,name,value):
  if type(self) is cls or name in ('num_warps','num_ctas','num_stages','num_buffers_warp_spec','num_consumer_groups','reg_dec_producer','reg_inc_consumer','maxnreg','cluster_dims','ptx_version','enable_fp_fusion','launch_cooperative_grid','supported_fp8_dtypes','deprecated_fp8_dtypes','default_dot_input_precision','allowed_dot_input_precisions','max_num_imprecise_acc_default','extern_libs','debug','backend_name','sanitize_overflow','arch',):
   raise FrozenInstanceError(f"cannot assign to field {name!r}")
  super(cls, self).__setattr__(name, value)
 return __setattr__