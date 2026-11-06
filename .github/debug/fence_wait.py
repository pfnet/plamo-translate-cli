"""A GPU op that consumes a slow CPU-stream result: evaluated together, then CPU first."""
import sys
import time

import mlx.core as mx


def slow_cpu(seconds_hint: int) -> mx.array:
    a = mx.random.normal((2048, 2048), stream=mx.cpu)
    for _ in range(seconds_hint):
        a = mx.matmul(a, a, stream=mx.cpu) / 64
    return a


mode = sys.argv[1]
steps = int(sys.argv[2])
cpu = slow_cpu(steps)
start = time.perf_counter()
try:
    if mode == "cpu-first":
        mx.eval(cpu)
    gpu = (cpu * 2).sum()  # default stream: GPU
    mx.eval(gpu)
    print(f"{mode} steps={steps}: ok in {time.perf_counter() - start:.1f} s")
except RuntimeError as exc:
    print(f"{mode} steps={steps}: {exc} after {time.perf_counter() - start:.1f} s")
