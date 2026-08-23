# Two-B200 execution plan

Two B200-class GPUs are enough to keep one complete Qwen3-30B-A3B BF16 replica
on each GPU and split independent windows/candidate sweeps. This is preferred to
tensor parallelism for the 30.5B/3.3B-active pilot unless a single experiment's
activation memory proves larger than one GPU.

## Pre-rental gate

Verify on the rental before transferring large artifacts:

```bash
nvidia-smi --query-gpu=name,memory.total,compute_cap,pci.bus_id --format=csv
python3 -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability())'
nvcc --version
```

B200 is an SM100 target. The local workstation extension and runtime images are
SM120-specific and must not be reused. For a direct PyTorch extension build set
`TORCH_CUDA_ARCH_LIST=10.0`. If using the repository's generalized
`bootstrap_ext_sm120.py` helper, set its wrapper input `EXL3_ARCH_LIST=10.0`;
the helper validates that value and exports it as `TORCH_CUDA_ARCH_LIST` before
compilation. Verify the resulting binary actually contains the required
encode/reconstruct ops and passes a small on-device round trip before the paid
campaign. Serving kernels are a separate qualification and may need an
SM100-native image.

## Work split

- GPU 0: even-numbered sealed windows; primary codec/objective arms.
- GPU 1: odd-numbered windows; controls and joint-triplet spot checks.
- Merge only immutable per-window receipts; reject overlapping window IDs.
- Run exact final KLD only after candidate hashes are frozen.

A practical first reservation is 8-12 hours, with artifacts written to durable
storage continuously. Extend only after the SM100 extension smoke, BF16 logit
parity, and expected throughput pass. The full wall time will be dominated by
the actual MCG candidate encodes and path-node backpropagations, not model fit.
