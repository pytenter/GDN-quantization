# Bottleneck ranking

Top-level inclusive ranking:

1. **Python_reducer** — 5494.240 ms/token (93.81% of observed wall)
2. **forward** — 231.063 ms/token (3.95% of observed wall)
3. **GPU_to_CPU_transfer** — 26.338 ms/token (0.45% of observed wall)
4. **quantizer** — 20.400 ms/token (0.35% of observed wall)

Reducer subcomponent ranking:

1. **histogram_moments** — 4123.584 ms/token (70.40% of observed wall)
2. **TopK** — 1223.438 ms/token (20.89% of observed wall)
3. **reducer_other** — 74.310 ms/token (1.27% of observed wall)
4. **tails** — 54.911 ms/token (0.94% of observed wall)
5. **paired_counts** — 17.998 ms/token (0.31% of observed wall)

`PYTHON_REDUCER_DOMINANT = YES`
`TRANSFER_DOMINANT = NO`
`TAIL_DOMINANT = NO`
`FORWARD_DOMINANT = NO`

Subcomponent percentages are inclusive and must not be summed with their parent component.
