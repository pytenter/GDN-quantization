def __create_fn__():
 def __repr__(self):
  return self.__class__.__qualname__ + f"(load_in_8bit={self.load_in_8bit!r}, llm_int8_threshold={self.llm_int8_threshold!r}, load_in_4bit={self.load_in_4bit!r}, bnb_4bit_quant_type={self.bnb_4bit_quant_type!r}, bnb_4bit_use_double_quant={self.bnb_4bit_use_double_quant!r}, bnb_4bit_compute_dtype={self.bnb_4bit_compute_dtype!r}, torch_dtype={self.torch_dtype!r}, skip_modules={self.skip_modules!r}, keep_in_fp32_modules={self.keep_in_fp32_modules!r})"
 return __repr__