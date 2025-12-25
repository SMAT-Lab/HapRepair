# Preliminary 实验说明（Codex 缺陷检测能力评估）

本文件说明当前 `revision/` 目录下和 preliminary 实验相关的代码、数据与使用方式，主要用于支撑一个结论：

> 将 LLM（这里是通过 Codex CLI 调用的大模型）直接当作 ArkTS 性能缺陷检测器，在真实项目上的召回率和稳定性都不够，尤其是在单文件多缺陷场景，从而需要一个类似 HomeCheck 的静态分析前端。

## 数据集与 Ground Truth

- 选取的数据集来自 OpenHarmony 官方 XTS ArkUI 列表模块：
  - 源代码路径（已拷贝到 revision）：  
    - `revision/ace_ets_module_list02/`
  - 原始项目来源：  
    - `/home/LLMCodeRepair/repo_new/OpenHarmony/xts_acts/arkui/ace_ets_module_ui/ace_ets_module_scroll/ace_ets_module_list02`

- 可选参考：对该模块运行 CodeLinter 得到缺陷报告（用于交叉核对规则含义与触发位置）：
  - 日志文件（仅说明来源，不在 revision 中作为 GT 直接使用）：  
    - `/home/LLMCodeRepair/logs/codelinter_openharmony/ace_ets_module_list02.log`

- 在 `revision/` 中我们根据规则描述与源码检查**人工构造**了 JSON 格式的 ground truth：
  - `revision/ace_ets_module_list02_gt.json`
  - 结构（按相对路径组织，行号精确到单个缺陷）：
    ```json
    {
      "entry/src/main/ets/MainAbility/pages/List/List07.ets": {
        "defects": [
          {
            "line": 44,
            "severity": "suggestion",
            "category": "performance",
            "rule": "@performance/hp-arkui-remove-unchanged-state-var"
          },
          {
            "line": 63,
            "severity": "warn",
            "category": "performance",
            "rule": "@performance/hp-arkui-load-on-demand"
          },
          ...
        ],
        "rules": [
          "@performance/hp-arkui-load-on-demand",
          "@performance/hp-arkui-remove-unchanged-state-var"
        ]
      },
      ...
    }
    ```

在本文实验中，“一个缺陷”指一个 `(rule, line)` 对，即特定文件中某条规则在某一行的触发点。

## 检测与评估代码

### 1. Codex 检测 + 在线评估脚本

- 路径：  
  - `revision/code/detect_and_eval_codex.py`

- 作用：
  1. 遍历 GT 中列出的所有 ArkTS 文件（`ace_ets_module_list02_gt.json`），构造检测 prompt；
  2. 通过 `agent/run_codex.py` 调用 Codex CLI，让大模型输出预测的缺陷列表；
  3. 解析 Codex 输出，并与 GT 做对齐；
  4. 以**缺陷级别（rule + line）**统计整体检测效果，并区分：
     - 单缺陷文件（该文件 GT 中只有 1 个缺陷）；
     - 多缺陷文件（GT 中有多个缺陷）。

- 使用方式（会实际调用 Codex，耗时取决于模型和网络）：
  ```bash
  cd /home/LLMCodeRepair/revision
  python code/detect_and_eval_codex.py
  ```

- 输出与结果存储：
  - 终端输出：打印每个文件的 GT 与预测规则，以及整体统计：
    - 总 GT 缺陷数 / 总预测缺陷数 / 正确检测缺陷数；
    - 单缺陷文件 vs 多缺陷文件的同类统计。
  - 结果目录：
    - 稳定目录（用于 resume 和评估）：  
      - `revision/ace_ets_module_list02_codex_eval/latest/`
        - `per_file_results.ndjson`：一行一个文件的 JSON 记录，包含：
          - `filename`：相对路径；
          - `gt_defects`：`[{ "rule": ..., "line": ... }, ...]`；
          - `pred_defects`：`[{ "rule": ..., "line": ... }, ...]`；
          - `parse_ok` / `raw_output` / `trace`。
        - `summary.json` / `per_file_results.json`：完整快照。
    - 时间戳快照目录（每次运行一份，不用于 resume）：  
      - `revision/ace_ets_module_list02_codex_eval/<timestamp>/`

- 并发与 resume：
  - 默认使用 8 线程并发调用 Codex（`ThreadPoolExecutor(max_workers=8)`）。
  - 再次运行脚本时，会读取 `latest/per_file_results.ndjson` 中已有文件的结果，并跳过这些文件（打印 `[resume] Skip ...`），只对未完成的文件调用 Codex。

### 2. 纯评估脚本（不重跑 Codex）

- 路径：  
  - `revision/code/eval_codex_results.py`

- 作用：
  - 只从以下两个文件中读取数据，**不再调用 Codex**：
    - GT：`revision/ace_ets_module_list02_gt.json`
    - 预测结果：`revision/ace_ets_module_list02_codex_eval/latest/per_file_results.ndjson`
  - 以缺陷级别 `(rule, line)` 对齐 GT 与预测：
    - 统计：
      - `GT 缺陷数`；
      - `预测缺陷数`（Codex 报出的所有 `(rule, line)`）；
      - `正确检测的缺陷数`（两者交集）。
    - 分别给出：
      - 总体；
      - 单缺陷文件；
      - 多缺陷文件。

- 使用方式：
  ```bash
  cd /home/LLMCodeRepair/revision
  python code/eval_codex_results.py
  ```

- 示例输出含义：
  ```text
  === Defect-level evaluation (using latest Codex results) ===
  Files (GT): 18  Single-defect files: 6,  Multi-defect files: 12

  -- Overall --
  GT defects     : 38          # GT 中一共 38 个缺陷（rule+line）
  Predicted      : 43          # Codex 一共报了 43 个缺陷
  Correctly found: 25          # 其中有 25 个刚好命中 GT
  Coverage       : 0.6579      # 召回率 = 25 / 38
  Precision      : 0.5814      # 精度 = 25 / 43
  F1             : 0.6173

  -- Single-defect files --
  GT defects     : 6
  Predicted      : 11
  Correctly found: 6           # 单缺陷文件上 recall=1.0，但预测多报了一些

  -- Multi-defect files --
  GT defects     : 32
  Predicted      : 32
  Correctly found: 19          # 多缺陷文件上只检出了 ~59% 的缺陷
  ```

## 实验结论在论文中的用法建议

基于上述脚本与数据，本 preliminary 实验主要支持以下观察：

- 即便在 OpenHarmony 官方 XTS 模块这种“结构清晰、规则明确”的场景中，LLM 直接作为缺陷检测器：
  - **单缺陷文件**上可以找到所有缺陷，但会多报不少（需要人工筛选）；
  - **多缺陷文件**上会稳定漏掉约 40% 的真实缺陷（Coverage ~0.6），且预测数与 GT 数量接近，并非通过“多报”换来高召回率。
- 因此，将 LLM 作为“前端检测器”在工业场景下难以获得稳定的高召回率和高精度；
- 这就为后续引入 HomeCheck 作为高精度静态分析前端（负责定位缺陷），而让 LLM 专注于基于告警进行修复（HapRepair），提供了合理的动机。

如果未来需要扩展 preliminary 实验，可以：

- 从 `logs/codelinter_openharmony*` 中再挑选 1～2 个类似规模的项目；
- 使用完全相同的 pipeline（GT JSON + Codex detection + eval_codex_results.py）复现检测效果；
- 将多项目的结果整理在论文附录或 Threats to Validity 中，以增强说服力。
