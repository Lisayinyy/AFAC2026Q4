#!/usr/bin/env bash
# 一键打包 submission.zip（B 榜前 15 代码审核要求的结构）
set -e
cd "$(dirname "$0")/.."
[ -f output/answer.csv ] || { echo "缺 output/answer.csv，先跑 script/run_a.py"; exit 1; }
rm -rf submission submission.zip
mkdir -p submission/script submission/logs
cp output/answer.csv output/evidence.json submission/
cp -r agent submission/agent
cp script/*.py submission/script/
cp output/token_ledger.json submission/logs/ 2>/dev/null || true
cp output/progress.jsonl submission/logs/ 2>/dev/null || true
cp requirements.txt README.md submission/
zip -qr submission.zip submission
echo "生成 submission.zip ($(du -h submission.zip | cut -f1))"
echo "线上评测只需上传 output/answer.csv"
