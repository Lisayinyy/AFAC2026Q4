# AFAC2026 · 金融长文本 Agent 的动态记忆压缩与高效问答

赛题链接：https://tianchi.aliyun.com/competition/entrance/532486/information

## 赛题简介

AFAC2026 金融智能创新大赛「挑战组」赛题四。主办方提供海量、结构复杂的金融长文档（年报、保险条款、募集说明书、监管法规、研究报告等），选手需构建一个 Agent，在**控制 Token 成本**的前提下，基于这些长上下文对给定问题做出**精准问答**。

核心难点：

- **文档结构复杂**：大量交叉引用、多级标题、密集表格、附录与批注，一个否定词或一个单元格错位就会导致答案错误。
- **上下文超长**：单份文档动辄上百页，无法整体塞入模型窗口，需要检索 / 分块 / 动态记忆压缩等工程手段。
- **成本约束**：在保证准确率的同时优化 Token 消耗，考验召回与压缩策略的取舍。

## 数据集结构

数据集位于同级目录 `../public_dataset_upload/`：

```
public_dataset_upload/
├── questions/
│   └── group_a/                       # A 榜题目（共 100 题，每个领域 20 题）
│       ├── insurance_questions.json
│       ├── financial_reports_questions.json
│       ├── financial_contracts_questions.json
│       ├── regulatory_questions.json
│       └── research_questions.json
└── raw/                               # 题目引用的原始文档
    ├── insurance/                     # 保险产品条款 PDF（1.pdf ~ 16.pdf）
    ├── financial_reports/             # 上市公司年报 PDF（比亚迪/宁德/中建/招行/美的等）
    ├── financial_contracts/           # 债券募集说明书等合同 PDF（text01 ~ text14）
    ├── research/                      # 行业研究报告 PDF（pack2_text01 ~ pack2_text20）
    └── regulatory/                    # 监管法规
        ├── txt/                       # 法规正文（6 篇 .txt）
        └── attachments/               # 法规附件 PDF（csrc_xxxx_attN.pdf，共 130 个）
```

### 五个业务领域

| 领域 | 目录 | 文档类型 | 题目文件 |
|------|------|----------|----------|
| 保险 insurance | `raw/insurance` | 养老 / 寿险 / 医疗险产品条款 | `insurance_questions.json` |
| 财报 financial_reports | `raw/financial_reports` | 上市公司年度报告 | `financial_reports_questions.json` |
| 金融合同 financial_contracts | `raw/financial_contracts` | 债券募集说明书等 | `financial_contracts_questions.json` |
| 监管 regulatory | `raw/regulatory` | 央行 / 金监总局 / 证监会法规 | `regulatory_questions.json` |
| 研究报告 research | `raw/research` | 行业研究报告 | `research_questions.json` |

## 题目格式

每道题为一个 JSON 对象，示例：

```json
{
  "qid": "fin_a_001",
  "domain": "financial_reports",
  "split": "A",
  "question": "根据比亚迪连续两年的年度报告，下列关于公司经营业绩变化的描述中，哪些是准确的？",
  "options": { "A": "...", "B": "...", "C": "...", "D": "..." },
  "answer_format": "multi",
  "type": "财务指标对比分析",
  "doc_ids": ["annual_byd_2024_report", "annual_byd_2025_report"]
}
```

字段说明：

- `qid`：题目唯一 ID。
- `domain`：所属领域（对应上表）。
- `question` / `options`：题干与选项。
- `answer_format`：答案类型，取值见下表。
- `type`：题目细分类型（如计算题、推理判断、财务指标对比等）。
- `doc_ids`：该题引用的原始文档标识，对应 `raw/<domain>/` 下的文件名（不含扩展名）。

### 答案类型分布（A 榜 100 题）

| answer_format | 含义 | 数量 |
|---------------|------|------|
| `multi` | 多选题（多个正确选项） | 65 |
| `tf` | 判断题（正确 / 错误） | 20 |
| `mcq` | 单选题 | 15 |

> 注：公开数据集中的题目**不含标准答案**，答案由官方评测时持有。

## 97→100 的回归与审计

最新 Agent 使用选项级证据矩阵。为减少最后几题常见的偶发错误，本分支额外加入：

- 标题/语义边界切块与带版本缓存，避免旧缓存掩盖解析改动；
- 金融数字表达规范化（千分位、数字空格、中文百分比）；
- 区域感知 MMR，避免同一章节的重叠块挤掉第二条关键证据；
- 表格行和句子边界安全截断；
- 证据引用校验，以及首判/复核的置信度门控合并。

### 细粒度证据层

保险文档已从单层 Chunk 检索升级为四层结构：

```text
DocumentNode -> SectionNode -> EvidenceGroup -> AtomicFact
```

- BM25 建在带产品身份、章节和事实类型的 `AtomicFact` 上；
- 命中后按 `group_id` 去重并展开到完整 `EvidenceGroup`，避免条件和结论断裂；
- 保险事实标记为责任、免责、等待期、宽限期、合同状态、免赔/公式等类型；
- 高区分度条款词在目标产品中缺失时，不再从其他产品借用条款；
- 对“不涵盖某费用”这类封闭责任清单命题，会回退检索该产品自身的保险责任清单；
- 原始 Chunk 仍保留用于文档身份卡和全文词频，避免原子化造成重复统计。

数据结构位于 `src/schema.py`，保险切分器位于 `src/segment/insurance.py`。

财报表格使用同一层级模型，切分器位于 `src/segment/financial_report.py`：

- PDF 提取时保留表格页码、表格上方标题、单位及口径；
- 超长表格跨块时重复年份表头；相邻页续表继承上一页年份表头；
- 每个财务指标行独立索引，并保存年份到数值的结构化映射；
- 查询中的“营收/营业总收入/归母净利润”等简称会映射到披露指标名；
- 跨年度同名指标不参与 MMR 重复惩罚，确保比较题同时获得两年数据；
- 财报默认每选项保留 6 个证据组，低覆盖扩检最多 8 个。

法规文档由 `src/segment/regulatory.py` 处理：

- 以章、条、款/项建立父子关系，命中“（一）”时自动补回所属完整条文；
- 同一旧 Chunk 内包含多个条文时会再次按“第X条”拆分，避免条号错配；
- 区分正式规则、案例事实、当事人申辩和监管认定，规则题优先正式条文；
- 非条文化的处罚决定按原始段落建立证据组，防止整份决定书形成超长父块；
- 法规默认每选项保留 6 个证据组，低覆盖扩检最多 8 个。

条件 Review 改为风险排序：高置信且引用完整的首判不复核；多选题只保留一个最有
证据基础的漏选申诉，正常题最多复核两个选项。复核检索只发送相对首轮新增的最多
5 个证据组，不再重复发送整份选项证据矩阵。

离线回归：

```bash
python -m pytest -q
```

拿到与官网 97 题正确记录一一对应的提交文件后，可与历史 95 分提交做约束比较：

```bash
python src/compare_scored_runs.py \
  --run baseline95:output/answer_group_a.csv:95 \
  --run agent97:output/answer_agent_97.csv:97
```

工具会校验两份文件和分数是否相容，并列出分歧题中新版必然修对、改错和共同仍错
的数量。它不会把 97 分文件当作标准答案，也不会凭分数猜测具体标签。
