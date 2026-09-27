"""Central configuration: paths and pipeline hyper-parameters.

All paths can be overridden with environment variables so the same code runs
locally and on Colab:
    BER_DATA_DIR  -> folder containing train/ and test/   (default: ../../../dataset)
    BER_WORK_DIR  -> folder for intermediate parquet/models (default: ../../../work)
    BER_OUT_DIR   -> folder for the two submission files   (default: ../../../output)
"""
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]  # student_resource/

DATA_DIR = Path(os.environ.get("BER_DATA_DIR", _ROOT / "dataset"))
WORK_DIR = Path(os.environ.get("BER_WORK_DIR", _ROOT / "work"))
OUT_DIR = Path(os.environ.get("BER_OUT_DIR", _ROOT / "output"))
for _d in (WORK_DIR, OUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42

# ---------------- blocking ----------------
# Keys whose document frequency on the candidate side exceeds the cap are
# dropped (too generic to be useful and they explode the join).  Caps are
# expressed for the full-size data; DEV runs scale them by BLOCK_SCALE.
BLOCK_SCALE = float(os.environ.get("BER_BLOCK_SCALE", "1.0"))
_CM = float(os.environ.get("BER_CAP_MULT", "1.0"))   # >1 = keep more common keys (higher recall, slower)
CAP_NAME_TOKEN = int(300 * _CM)
CAP_NAME_PAIR = int(300 * _CM)
CAP_NAME_CONCAT = int(100 * _CM)
CAP_ADDR_PAIR = int(300 * _CM)
CAP_ADDR_TOKEN = int(300 * _CM)
CAP_MIXED = int(300 * _CM)
CAP_EXACT = int(1000 * _CM)
S3_STRONG = os.environ.get("BER_S3_STRONG", "0") == "1"   # bigger/slower stage-3 LightGBM
BLOCK_CHUNK = int(os.environ.get("BER_BLOCK_CHUNK", "100000"))   # S1 records per blocking batch
RERANK_M = int(os.environ.get("BER_RERANK_M", "120"))            # shortlist depth re-scored before top-K
BLOCK_TOPK = int(os.environ.get("BER_BLOCK_TOPK", "25"))   # stage-1 shortlist per S1

# ---------------- pruning / final candidate set ----------------
PRUNE_MAX = 12          # hard cap of candidates per S1 after the pruner
PRUNE_MIN_PROB = 0.02   # pruner probability below which a candidate is dropped

N_FOLDS = 4

# Test has ~5.8 S2/S3 records per S1 entity vs ~4.7 in train (=> ~40% vs ~26% unmatched "distractor"
# records).  Dropping this fraction of train S1 entities (their S2/S3 records stay and become
# distractors) makes the training distribution match the test distribution.
TRAIN_S1_KEEP = float(os.environ.get("BER_TRAIN_S1_KEEP", "0.8"))
