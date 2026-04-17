# Torchrun NCCL EP Benchmark

This directory contains a `torchrun`-based replacement for the MPI-launched
`contrib/nccl_ep/ep_bench.cu` benchmark.

The implementation is split in two parts:

- `ep_bench_core.cu`: MPI-free C++ benchmark core exposed via `ep_bench_run(...)`
- `ep_bench.py`: Python launcher that uses `torchrun` + `torch.distributed` (gloo control plane)

## Build

Build NCCL and `libnccl_ep.so` first, then build the local benchmark core:

```bash
export NCCL_HOME=/path/to/nccl/build
make -C contrib/nccl_ep/ep_bench_python
```

If you built NCCL into the repo default location, `NCCL_HOME` is optional and
defaults to `./build`.

## Environment

At runtime, the benchmark needs the NCCL EP libraries on the loader path:

```bash
export NCCL_HOME=/path/to/nccl/build
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$LD_LIBRARY_PATH
```

If PyTorch would otherwise pick a different NCCL, preload the one from
`$NCCL_HOME` before starting Python:

```bash
export LD_PRELOAD=$NCCL_HOME/lib/libnccl.so.2${LD_PRELOAD:+:$LD_PRELOAD}
```

## Run

Example LL run:

```bash
torchrun --standalone --nproc_per_node=8 \
  contrib/nccl_ep/ep_bench_python/ep_bench.py \
  --algorithm ll --tokens 128 --hidden 7168 --top-k 8 --experts 256
```

Example HT run:

```bash
torchrun --standalone --nproc_per_node=8 \
  contrib/nccl_ep/ep_bench_python/ep_bench.py \
  --algorithm ht --tokens 4096 --hidden 7168 --top-k 8 --experts 256
```

Profiling example:

```bash
nsys profile --stats=true \
  torchrun --standalone --nproc_per_node=8 \
  contrib/nccl_ep/ep_bench_python/ep_bench.py --algorithm ll --profile
```

## Notes

- `torchrun` is the only supported launcher.
- The Python control plane uses `gloo`; the benchmark data plane still uses the
  NCCL communicator created inside `ep_bench_core.cu`.
- `--dynamic-tokens` is still intentionally unsupported, matching the original
  benchmark behavior.
