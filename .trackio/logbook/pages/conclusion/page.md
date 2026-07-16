# Conclusion


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_42b048d1fad8", "created_at": "2026-07-16T16:30:20+00:00", "title": "Final outcomes"}
-->
# Conclusion

- **Claim 1: VERIFIED.** Adam is linear while GD and momentum recover their predicted sublinear laws.
- **Claim 2: VERIFIED.** Second-moment memory decouples from the shrinking instantaneous gradient and exponentially amplifies the effective learning rate.

## Scope & cost

| | Scope | Hardware | Time | Cost | Outcome |
|---|---|---|---:|---:|---|
| This reproduction | k=4,6,8,10; 11 precision levels; 8 slow controls; RMSProp | local CPU | ~4 s | $0 | both claims verified |
| Full replication | same scalar theorem family | CPU | seconds | $0 | fully covered at and beyond plotted degrees |
