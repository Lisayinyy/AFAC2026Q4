# AFAC2026 金融长文本 Agent - 全局配置
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("AFAC_DATA_ROOT", REPO_ROOT.parent / "public_dataset_upload"))
QUESTIONS_DIR = DATA_ROOT / "questions" / "group_a"
RAW_DIR = DATA_ROOT / "raw"
PROCESSED_DIR = REPO_ROOT / "processed_data"
OUTPUT_DIR = REPO_ROOT / "output"

DOMAINS = ["insurance", "financial_reports", "financial_contracts", "regulatory", "research"]

# ---- 模型配置：仅允许 Qwen 系列（百炼 OpenAI 兼容接口）----
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY_ENV = "DASHSCOPE_API_KEY"

# 答题主模型（评测基准为 qwen3.6-plus，以百炼实际模型名为准，可用环境变量覆盖）
ANSWER_MODEL = os.environ.get("AFAC_ANSWER_MODEL", "qwen-plus")
# 轻量模型：用于证据段落筛选/文档路由，省 token
LITE_MODEL = os.environ.get("AFAC_LITE_MODEL", "qwen-turbo")

# ---- Token 预算 ----
TOKEN_BUDGET_TOTAL = 5_000_000  # 官方 TokenBudget
# 单题目标预算（软约束，用于控制检索段数量）
PER_QUESTION_SOFT_BUDGET = 30_000

# ---- 检索参数 ----
CHUNK_SIZE = 800          # 字符数
CHUNK_OVERLAP = 150
TOP_K_CHUNKS = 8          # 每题送入答题模型的证据块数
BM25_CANDIDATES = 30      # BM25 初筛数量（再由 LITE_MODEL 精选时使用）
