# GDN INT8 Orientation Trajectory Source Audit V1

## OBSERVATION
- continuation_source: `P0_FP_STATE_DECODED_RESPONSE_RETOKENIZED`
- exact_original_generation_token_ids_available: `false`
- exact replay claimed: `false`

## RECORDS
```json
[
  {
    "continuation_source": "P0_FP_STATE_DECODED_RESPONSE_RETOKENIZED",
    "decode_encode_skip_special_text_exact_match": true,
    "decode_encode_text_exact_match": true,
    "exact_original_generation_token_ids_available": false,
    "fp_correct": true,
    "fp_tokens_reported_by_p0": 1909,
    "fp_truncated": false,
    "int8_column_correct": true,
    "int8_column_truncated": false,
    "int8_row_correct": false,
    "int8_row_truncated": true,
    "normalization_difference": {
      "decoded_len": 4965,
      "decoded_sha256": "d8d055d8919a7d9e755e5a2fd25559f497be9d389387d8caf3f547356bafdb00",
      "decoded_skip_special_len": 4965,
      "decoded_skip_special_sha256": "d8d055d8919a7d9e755e5a2fd25559f497be9d389387d8caf3f547356bafdb00",
      "leading_changed": false,
      "original_len": 4965,
      "original_sha256": "d8d055d8919a7d9e755e5a2fd25559f497be9d389387d8caf3f547356bafdb00",
      "trailing_changed": false
    },
    "problem_id": "test/algebra/1332.json",
    "retokenized_token_count": 1908,
    "retokenized_with_special_token_count": 1908,
    "role": "INT8-row truncated pathological candidate",
    "special_token_delta": 0
  },
  {
    "continuation_source": "P0_FP_STATE_DECODED_RESPONSE_RETOKENIZED",
    "decode_encode_skip_special_text_exact_match": true,
    "decode_encode_text_exact_match": true,
    "exact_original_generation_token_ids_available": false,
    "fp_correct": true,
    "fp_tokens_reported_by_p0": 4536,
    "fp_truncated": false,
    "int8_column_correct": true,
    "int8_column_truncated": false,
    "int8_row_correct": false,
    "int8_row_truncated": true,
    "normalization_difference": {
      "decoded_len": 10522,
      "decoded_sha256": "84b5347f6b4d7e902cd4f0a8e083d91d35e61214b5bc3b4579ae54950660dd96",
      "decoded_skip_special_len": 10522,
      "decoded_skip_special_sha256": "84b5347f6b4d7e902cd4f0a8e083d91d35e61214b5bc3b4579ae54950660dd96",
      "leading_changed": false,
      "original_len": 10522,
      "original_sha256": "84b5347f6b4d7e902cd4f0a8e083d91d35e61214b5bc3b4579ae54950660dd96",
      "trailing_changed": false
    },
    "problem_id": "test/counting_and_probability/119.json",
    "retokenized_token_count": 4535,
    "retokenized_with_special_token_count": 4535,
    "role": "INT8-row truncated pathological candidate",
    "special_token_delta": 0
  },
  {
    "continuation_source": "P0_FP_STATE_DECODED_RESPONSE_RETOKENIZED",
    "decode_encode_skip_special_text_exact_match": true,
    "decode_encode_text_exact_match": true,
    "exact_original_generation_token_ids_available": false,
    "fp_correct": true,
    "fp_tokens_reported_by_p0": 5118,
    "fp_truncated": false,
    "int8_column_correct": true,
    "int8_column_truncated": false,
    "int8_row_correct": false,
    "int8_row_truncated": true,
    "normalization_difference": {
      "decoded_len": 18469,
      "decoded_sha256": "9c3786c25b08b1bc799c94deff698c1e7e498d901a1165140788027103533890",
      "decoded_skip_special_len": 18469,
      "decoded_skip_special_sha256": "9c3786c25b08b1bc799c94deff698c1e7e498d901a1165140788027103533890",
      "leading_changed": false,
      "original_len": 18469,
      "original_sha256": "9c3786c25b08b1bc799c94deff698c1e7e498d901a1165140788027103533890",
      "trailing_changed": false
    },
    "problem_id": "test/geometry/477.json",
    "retokenized_token_count": 5117,
    "retokenized_with_special_token_count": 5117,
    "role": "INT8-row truncated pathological candidate",
    "special_token_delta": 0
  },
  {
    "continuation_source": "P0_FP_STATE_DECODED_RESPONSE_RETOKENIZED",
    "decode_encode_skip_special_text_exact_match": true,
    "decode_encode_text_exact_match": true,
    "exact_original_generation_token_ids_available": false,
    "fp_correct": true,
    "fp_tokens_reported_by_p0": 18813,
    "fp_truncated": false,
    "int8_column_correct": true,
    "int8_column_truncated": false,
    "int8_row_correct": false,
    "int8_row_truncated": true,
    "normalization_difference": {
      "decoded_len": 43404,
      "decoded_sha256": "5024e4909a9e66fe9ba8ac8b756c40a454590143f60a77942efd17c2f99802b7",
      "decoded_skip_special_len": 43404,
      "decoded_skip_special_sha256": "5024e4909a9e66fe9ba8ac8b756c40a454590143f60a77942efd17c2f99802b7",
      "leading_changed": false,
      "original_len": 43404,
      "original_sha256": "5024e4909a9e66fe9ba8ac8b756c40a454590143f60a77942efd17c2f99802b7",
      "trailing_changed": false
    },
    "problem_id": "test/geometry/702.json",
    "retokenized_token_count": 18812,
    "retokenized_with_special_token_count": 18812,
    "role": "INT8-row truncated pathological candidate",
    "special_token_delta": 0
  },
  {
    "continuation_source": "P0_FP_STATE_DECODED_RESPONSE_RETOKENIZED",
    "decode_encode_skip_special_text_exact_match": true,
    "decode_encode_text_exact_match": true,
    "exact_original_generation_token_ids_available": false,
    "fp_correct": true,
    "fp_tokens_reported_by_p0": 914,
    "fp_truncated": false,
    "int8_column_correct": true,
    "int8_column_truncated": false,
    "int8_row_correct": false,
    "int8_row_truncated": false,
    "normalization_difference": {
      "decoded_len": 2511,
      "decoded_sha256": "7f4fe21da5f9c50ad392a463a1bd619a3a1314889c7b9adfdf9844a8c59cfdce",
      "decoded_skip_special_len": 2511,
      "decoded_skip_special_sha256": "7f4fe21da5f9c50ad392a463a1bd619a3a1314889c7b9adfdf9844a8c59cfdce",
      "leading_changed": false,
      "original_len": 2511,
      "original_sha256": "7f4fe21da5f9c50ad392a463a1bd619a3a1314889c7b9adfdf9844a8c59cfdce",
      "trailing_changed": false
    },
    "problem_id": "test/algebra/1214.json",
    "retokenized_token_count": 913,
    "retokenized_with_special_token_count": 913,
    "role": "INT8-row non-truncated termination control",
    "special_token_delta": 0
  },
  {
    "continuation_source": "P0_FP_STATE_DECODED_RESPONSE_RETOKENIZED",
    "decode_encode_skip_special_text_exact_match": true,
    "decode_encode_text_exact_match": true,
    "exact_original_generation_token_ids_available": false,
    "fp_correct": true,
    "fp_tokens_reported_by_p0": 5139,
    "fp_truncated": false,
    "int8_column_correct": true,
    "int8_column_truncated": false,
    "int8_row_correct": false,
    "int8_row_truncated": false,
    "normalization_difference": {
      "decoded_len": 11417,
      "decoded_sha256": "fa00fbc080a898a2074414265f9ee2dc0f5811f05771c0ed219624a8eba082f1",
      "decoded_skip_special_len": 11417,
      "decoded_skip_special_sha256": "fa00fbc080a898a2074414265f9ee2dc0f5811f05771c0ed219624a8eba082f1",
      "leading_changed": false,
      "original_len": 11417,
      "original_sha256": "fa00fbc080a898a2074414265f9ee2dc0f5811f05771c0ed219624a8eba082f1",
      "trailing_changed": false
    },
    "problem_id": "test/intermediate_algebra/207.json",
    "retokenized_token_count": 5138,
    "retokenized_with_special_token_count": 5138,
    "role": "INT8-row non-truncated termination control",
    "special_token_delta": 0
  }
]
```

## NEGATIVE RESULT
- Retokenized decoded P0 text is not labeled as exact replay of original generation token IDs.
