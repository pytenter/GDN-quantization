def __create_fn__(cls, FrozenInstanceError):
 def __setattr__(self,name,value):
  if type(self) is cls or name in ('num_warps','waves_per_eu','num_stages','num_ctas','num_buffers_warp_spec','num_consumer_groups','reg_dec_producer','reg_inc_consumer','extern_libs','cluster_dims','debug','sanitize_overflow','arch','supported_fp8_dtypes','deprecated_fp8_dtypes','default_dot_input_precision','allowed_dot_input_precisions','enable_fp_fusion','launch_cooperative_grid','matrix_instr_nonkdim','kpack','allow_flush_denorm','max_num_imprecise_acc_default','backend_name','instruction_sched_variant',):
   raise FrozenInstanceError(f"cannot assign to field {name!r}")
  super(cls, self).__setattr__(name, value)
 return __setattr__