# Two-B200 clean-node runbook — no campaign launch

This runbook stops after the read-only production preflight. It does not
authorize `campaign execute`, GPU forward passes, encoding, publication, SSH,
or Hub/GitHub writes. The owner must inspect the sealed plan and preflight
report and explicitly approve execution in a later turn.

The example configuration is intentionally incomplete. Every
`__REQUIRED_...__` value must be replaced with a locally verified path, hash,
budget, threshold, driver version, or service factory. Placeholder validation
fails closed.

## 1. Create an isolated environment

The target numeric environment is recorded in
`environments/b200-cu132.lock.json`; exact Python package pins are in
`environments/requirements-b200-cu132.txt`. The wheelhouse hash and host driver
remain required machine-specific seals.

```bash
export REPO_ROOT=/absolute/path/to/shapleymcg
export CAMPAIGN_ROOT=/absolute/durable/path/qwen3-30b-a3b-campaign
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_CUDA_ARCH_LIST=10.0
export TOKENIZERS_PARALLELISM=false

cd "$REPO_ROOT"
python3.12 -m venv .venv-b200
. .venv-b200/bin/activate
python -m pip install --no-index --find-links /absolute/path/to/sealed-wheelhouse \
  -r environments/requirements-b200-cu132.txt
python -m pip install --no-deps -e .
python scripts/verify_b200_environment.py \
  --lock environments/b200-cu132.lock.json
```

The final verifier command must be green without `--allow-placeholders`.

## 2. Prepare and verify source closures

### 2.1 Qwen BF16 checkpoint

The source checkpoint is pinned to Qwen commit
`1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9`. The preparer downloads exactly
the config, tokenizer, license, index, and 16 BF16 safetensor shards. It then
writes a sealed per-file source receipt containing the three hashes required by
`artifact-lock.json`: config, index, and canonical shard manifest.

```bash
# Preview only; no network access or destination creation:
python scripts/prepare_qwen_checkpoint.py \
  --destination /absolute/path/to/Qwen3-30B-A3B-Base

# Separate approval for the large network download and receipt creation:
python scripts/prepare_qwen_checkpoint.py \
  --destination /absolute/path/to/Qwen3-30B-A3B-Base \
  --receipt /absolute/durable/path/qwen-source-receipt.json \
  --execute
```

Copy the receipt's `config_sha256`, `index_sha256`, and
`shard_manifest_sha256` into the working artifact lock. Preserve the upstream
Qwen `LICENSE` with the checkpoint.

### 2.2 Corrected R10/EXL3 Python closure

The competitive codec adapter requires Brandon M. Music's corrected R10
`r7_encoder` Python closure and the pinned `encode_tr3_v31.py` numeric core.
They are published with the prior GLM-5.2 model. The repository includes a
per-file SHA256 manifest and a dry-run-by-default downloader pinned to HF commit
`7c73450f05a151439d0f184f216b1eefcc394a31`. The downloaded upstream license
is preserved and remains controlling for those files.

```bash
# Preview the exact repository, revision, paths, and file count:
python scripts/prepare_corrected_exl3_source.py \
  --destination /absolute/path/to/corrected-r10-source

# Separate approval for network download; every file is then hash-verified:
python scripts/prepare_corrected_exl3_source.py \
  --destination /absolute/path/to/corrected-r10-source \
  --execute

export CORRECTED_R10_ROOT=/absolute/path/to/corrected-r10-source/reproducibility/r10
export EXL3_NUMERIC_CORE="$CORRECTED_R10_ROOT/lineage/encode_tr3_v31.py"
```

Use `CORRECTED_R10_ROOT` as the adapter's corrected source root and
`EXL3_NUMERIC_CORE` as its numeric core. Do not substitute the similarly named
workspace-level historical encoder; the manifest binds the exact R10 closure
that contains `r7_encoder/r10_codec.py`.

### 2.3 B12X

The writer/reader closure is pinned to commit
`36bce2c1552ba2d47dc09f20a6f64fbfc8ec4ff8`. Preview first; the second command
is the only network-mutating checkout step.

```bash
python scripts/prepare_b12x_checkout.py \
  --destination /absolute/path/to/b12x-pinned

# Separate owner approval for network clone:
python scripts/prepare_b12x_checkout.py \
  --destination /absolute/path/to/b12x-pinned \
  --execute

python scripts/verify_b12x_checkout.py \
  --source /absolute/path/to/b12x-pinned \
  --require-clean
```

This pin proves the `btx-atoms-v1` format closure. The pinned B12X project
describes its kernels as SM120-only; it does **not** establish an SM100 serving
runtime. B200 runtime qualification remains a later independent gate. Do not
equate CPU checkpoint-reader success with native SM100 kernel support.

## 3. Build and smoke the corrected EXL3 encoder extension

The encoder extension is distinct from the B12X serving runtime. It is rebuilt
from ExLlamaV3 v0.0.43 commit
`c5d9c657966ffeeaa9353f0cc899f18629da4a13` against Torch 2.12.1+cu132 with
`TORCH_CUDA_ARCH_LIST=10.0`. Prepare a new clean checkout rather than relying on
an unsealed source archive.

```bash
# Preview, then separately approve the network checkout:
python scripts/prepare_exllamav3_checkout.py \
  --destination /absolute/path/to/exllamav3-v0.0.43
python scripts/prepare_exllamav3_checkout.py \
  --destination /absolute/path/to/exllamav3-v0.0.43 \
  --execute
python scripts/verify_exllamav3_checkout.py \
  --source /absolute/path/to/exllamav3-v0.0.43 \
  --require-clean

# Preview only:
python scripts/bootstrap_sm100_exl3.py \
  --source /absolute/path/to/exllamav3-v0.0.43

# Separate owner approval for compilation/install:
python scripts/bootstrap_sm100_exl3.py \
  --source /absolute/path/to/exllamav3-v0.0.43 \
  --max-jobs 32 \
  --execute

# Hash/closure preview; no extension import or GPU execution:
python scripts/smoke_sm100_stack.py \
  --extension /absolute/path/to/exllamav3_ext.so \
  --numeric-core "$EXL3_NUMERIC_CORE" \
  --b12x-source /absolute/path/to/b12x-pinned

# Separate owner approval for two-device on-GPU encoder smoke:
python scripts/smoke_sm100_stack.py \
  --extension /absolute/path/to/exllamav3_ext.so \
  --numeric-core "$EXL3_NUMERIC_CORE" \
  --b12x-source /absolute/path/to/b12x-pinned \
  --devices 0,1 \
  --execute
```

Seal the resulting extension SHA256 into both artifact and adapter inputs.

## 4. Estimate resources before copying the configuration

```bash
python scripts/estimate_qwen_b200_resources.py \
  --retention capture-plus-ledger \
  --fit-windows 32 \
  --selection-windows 16 \
  --confirmation-windows 16 \
  --final-windows 25 \
  --window-tokens 2048 \
  > /absolute/durable/path/qwen-resource-estimate.json
```

The estimator is conservative and exposes its formula components. Copy the
reviewed peak, reserve, VRAM, RAM, and CPU values into the adapter config; do
not silently lower them to fit a rental.

## 5. Resolve every placeholder and verify immutable inputs

### 5.1 Construct and seal the document corpus

The calibration input is UTF-8 JSONL with exactly one source document per
line. Every row must contain a stable `id`, a `domain`, and the full `text`:

```json
{"id":"document-0001","domain":"legal","text":"Complete source document ..."}
```

Supply enough full documents in at least four domains to produce 32 fit, 16
selection, 16 confirmation, and 25 final windows after target tokenization.
The sealer assigns whole documents—not fragments—to one role, deterministically
stratifies by domain, rejects duplicate IDs, and fails if any document crosses
roles. Record the input JSONL SHA256 before sealing.

```bash
quant-pipeline seal \
  /absolute/durable/path/qwen3-30b-a3b-b200/experiment.toml \
  --output /absolute/durable/path/qwen-sealed-corpus.json
```

Preserve both the input JSONL and sealed JSON. A public result is not fully
reproducible until those exact artifacts (or a lawful redistributable dataset
revision that regenerates them byte-for-byte) are published with their hashes.

### 5.2 Resolve the configuration

Copy `configs/qwen3-30b-a3b-b200` to a durable working directory. Replace every
placeholder, then hash the complete model index/shards, corpus, KLD window,
corrected encoder closure, built extension, environment wheelhouse, and pinned
B12X checkout.

```bash
python scripts/validate_repro_config.py \
  --config-dir /absolute/durable/path/qwen3-30b-a3b-b200
```

For review only, `--allow-placeholders` prints unresolved fields. It never
makes a placeholder production-valid.

## 6. Seal the plan

Planning hashes local files and provider source. It performs no preflight,
model load, forward pass, or encoding. It will fail until the concrete local
service factory and every input path exist.

```bash
mkdir -p "$CAMPAIGN_ROOT"
quant-pipeline campaign plan \
  --definition /absolute/durable/path/qwen3-30b-a3b-b200/campaign.json \
  --campaign-dir "$CAMPAIGN_ROOT" \
  --adapter quant_pipeline.campaign.qwen_adapter:QwenCampaignAdapter

quant-pipeline campaign status \
  --campaign-dir "$CAMPAIGN_ROOT" \
  --adapter quant_pipeline.campaign.qwen_adapter:QwenCampaignAdapter

quant-pipeline campaign audit \
  --campaign-dir "$CAMPAIGN_ROOT" \
  --adapter quant_pipeline.campaign.qwen_adapter:QwenCampaignAdapter
```

## 7. Run preflight only

The first command previews the operation. The second imports the sealed local
providers and checks files, environment, disks, CPUs, RAM, both SM100 GPUs,
free VRAM, BF16 support, and the corrected codec identity. It never calls
`CampaignRunner.execute()` or dispatches a stage.

```bash
python scripts/preflight_qwen_campaign.py \
  --campaign-dir "$CAMPAIGN_ROOT"

python scripts/preflight_qwen_campaign.py \
  --campaign-dir "$CAMPAIGN_ROOT" \
  --execute-preflight \
  > "$CAMPAIGN_ROOT/preflight-review.json"
```

Stop here. Review `plan.json`, the provider closure, artifact/environment
locks, resource estimate, and `preflight-review.json`.

## OWNER APPROVAL BARRIER

Do not run the following command merely because preflight passes:

```text
quant-pipeline campaign execute ... --execute
```

That command is intentionally omitted. A fresh, explicit owner approval is
required before anyone constructs or runs it.
