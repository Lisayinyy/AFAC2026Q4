# 提交冲刺清单（按 2026-07-07 时点）

## ⏰ 关键时间线（已非常紧迫）

| 节点 | 时间 | 状态 |
|---|---|---|
| **报名截止** | ~2026-07-09 早（详情页倒计时为准） | ❗未报名则一切无从谈起 |
| **A 榜最后提交** | 2026-07-10 23:59 | ❗须有有效 A 榜成绩才有 B 榜资格 |
| A 榜成绩公布 | 2026-07-13 | |
| **B 榜** | 2026-07-13 ~ 07-24 | 每交易日**必须**提交，≥8 个交易日 |
| B 榜成绩公布 | 2026-07-28 | |
| 报告提交（TOP15） | 2026-07-28 ~ 08-05 | |

## 每日提交操作（B 榜为每日必做）

```bash
# 1. 确认真实数据源可用（三选一环境变量）
export HUNDSUN_SDK_PATH=...   # 或 HUNDSUN_DSN / HUNDSUN_EXPORT_DIR

# 2. 收盘后拉当日数据并产出提交（T 日 23:59 前）
python run_batch.py --source hundsun \
  --stocks-file 股票样本.xlsx --dates $(date +%Y%m%d) --train

# 3. 自检清单
#    - output/<date>/ 下无 SYNTHETIC_DO_NOT_SUBMIT.txt（有=数据源没接上，禁止提交）
#    - batch_report.json 中 data_provenance != hundsun_FALLBACK_SYNTHETIC
#    - 100 行、代码带 .SH 后缀、日期为当日
# 4. 上传 output/<date>/submit.zip 到天池
```

## 🚫 合规红线

- **兜底合成数据的产物严禁提交**：评分基于真实行情，合成数据得分无效，且预测
  本质来自伪随机种子，会被认定为"随机填充"违规 → 取消成绩。
  `run_batch.py` 已加闸：auto 模式无真实源直接中止；`--allow-fallback` 仅供
  管线验证，产物目录会写入 `SYNTHETIC_DO_NOT_SUBMIT.txt` 标记。
- 禁按股票代码打标；识别必须来自 Level-2 派生特征。
- B 榜 TOP15 会被全链路复现审核：保持 main.py 可一键复现、init_env.sh 依赖完整。

## 当前待办（按优先级）

1. ❗**决策：是否报名**（截止 ~7/9；不报名则本项目仅作研究）
2. ❗接通真实恒生数据源（mcode 环境，三选一 env）→ 出 7/9、7/10 的 A 榜真数据提交
3. B 榜期间每日运行上面流程（可配 cron/定时任务）
4. 若进 TOP15：写 project_solution_report.docx + 打包复现代码
