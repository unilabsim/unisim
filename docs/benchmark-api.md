# Benchmark API Reservation

Version `0.1.x` reserves `BenchmarkCase` and `BenchmarkResult` as the stable
extension point for a future engine benchmark package. No workload runner,
timing loop, result comparison, or performance claim is implemented here.

Future benchmark work must fix scene/control/state semantics, schema version,
artifact digest, synchronization, and hardware/runtime provenance. It must
reuse the same `SimBackend` contract used by UniLab adapters.
