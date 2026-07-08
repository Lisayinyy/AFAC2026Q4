#!/usr/bin/env python3
# 预处理全部领域文档：PDF/txt -> processed_data/{domain}/{doc_id}.json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent import config
from agent.parse import preprocess_domain

if __name__ == "__main__":
    domains = sys.argv[1:] or config.DOMAINS
    force = "--force" in domains
    domains = [d for d in domains if not d.startswith("--")]
    for d in domains:
        print(f"=== 预处理 {d} ===")
        preprocess_domain(d, force=force)
