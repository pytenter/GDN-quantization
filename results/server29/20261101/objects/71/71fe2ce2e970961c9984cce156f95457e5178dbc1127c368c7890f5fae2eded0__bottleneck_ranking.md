# Bottleneck ranking

Top-level inclusive ranking:

1. **Python_reducer** — 6545.035 ms/token (94.76% of observed wall)
2. **forward** — 228.914 ms/token (3.31% of observed wall)
3. **GPU_to_CPU_transfer** — 26.971 ms/token (0.39% of observed wall)
4. **quantizer** — 20.391 ms/token (0.30% of observed wall)

`PYTHON_REDUCER_DOMINANT = YES`
`TRANSFER_DOMINANT = NO`
`TAIL_DOMINANT = YES`
`FORWARD_DOMINANT = NO`

Subcomponent percentages are inclusive and must not be summed with their parent component.
