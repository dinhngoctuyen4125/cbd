# CBD

Code for reproducing CBD experiments on the Deepseek dataset.

## Setup

```bash
conda env create -f environment.yaml
conda activate cbd_agv
```

## Data

Place the dataset directory `deepseek/` (containing `D_forget.json`, `D_test_U_dep.json`, `D_test_U_nondep.json`) at `../Data-Collection/deepseek/` relative to this repo.

## Run

Run the full pipeline (basis extraction → training → threshold → scoring):

```bash
bash run_script.sh
```

To run in background:

```bash
mkdir -p logs
nohup bash run_script.sh > logs/pipeline.log 2>&1 &
tail -f logs/pipeline.log
```

### Pipeline Stages

| Stage | Script | Description |
|-------|--------|-------------|
| 1 | `extract_cbd_dfb_basis.py` | Extract CBD-DFB basis Q from gradients |
| 2 | `hf_forget_train.py` | Train model A1 (unlearning via GD+KL) |
| 3 | `assis_tinyllama_test_path.py` + `analyze_cross_entropy.py` | Compute sym-KL threshold (200 samples/set) |
| 4 | `assis_tinyllama_test_path.py` + `analyze_cross_entropy.py` | Score remaining samples & statistics |

### Configuration

Edit the top of `run_script.sh` to change parameters:

```bash
ASSIST_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEEPSEEK_DATA="../Data-Collection/deepseek"
MAX_LEN=512
TOP_K=192
THRESHOLD_SAMPLES=200
SCORE_SAMPLES=500
```

### Output

Results are saved to:
- `artifacts/basis_cbd_dfb/deepseek/` — Basis Q
- `artifacts/outputs_trained_models/` — Trained model checkpoints
- `artifacts/ce_deepseek/threshold/` — Threshold results
- `artifacts/ce_deepseek/scoring/` — Scoring results
