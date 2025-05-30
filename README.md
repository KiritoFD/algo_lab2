# DNA序列锚点链匹配实验报告

## 一、实验任务

本实验旨在实现一个高效的算法，用于比对两条DNA序列（一条参照序列 Reference, 一条查询序列 Query），并输出它们之间最优的匹配关系。这种匹配关系通过一系列非重叠的“锚点链”（Chains of Anchors）来表示。锚点是两条序列中高度相似或完全相同的短片段（如k-mers）。

**输入**:
1.  参照DNA序列 (Reference Sequence)
2.  查询DNA序列 (Query Sequence)

**输出**:
1.  一个表示最优匹配关系的字符串，格式为 `q_start,q_end,r_start,r_end,...`，其中 `q_start, q_end` 是查询序列上的起始终止坐标，`r_start, r_end` 是参照序列上的起始终止坐标。这些坐标对定义了一系列非重叠的对齐区域。

**要求**:
1.  算法的核心部分时间复杂度不应高于平方量级 O(n²)，其中n是锚点数量。
2.  算法设计中需体现图相关的算法思想。
3.  代码实现需上传到GitHub。
4.  提交一份实验文档（本README文件），包含算法描述、伪代码、时空复杂度分析、代码实现说明、参数调优说明及运行结果。
5.  最终的匹配结果需实名提交至指定的评分网站。

---

## 二、算法设计与伪代码

本算法将锚点链问题建模为在有向无环图（DAG）中寻找最长加权路径的问题。

1.  **锚点生成**：首先，通过比较参照序列和查询序列，找出所有匹配的k-mer对（锚点）。每个锚点记录其在查询序列和参照序列上的起始位置、链方向（正向/反向）以及k-mer长度。这一步由 `eval.py` 中的 `seq2hashtable_multi_test` 函数完成。
2.  **图构建（概念性）**：
    *   将每个锚点视为图中的一个节点。
    *   如果两个锚点 `j` 和 `i` (假设 `j` 在 `i` 之前，基于排序) 可以在生物学上合理地连接成链（即它们在查询序列和参照序列上都保持一致的顺序，满足最大间隙、最大对角线差异和允许的重叠等约束），则从节点 `j` 向节点 `i` 画一条有向边。边的权重可以是锚点 `i` 的长度（k-mer size）或其它评分。
3.  **动态规划求解**：
    *   **锚点排序**：为了高效地进行动态规划，首先将所有锚点主要按查询序列起始位置 (`q_s`) 升序排序，若 `q_s` 相同，则按参照序列起始位置 (`r_s`) 升序排序。
    *   **DP状态定义**：`dp[i]` 表示以排序后的第 `i` 个锚点结尾的最优锚点链的总得分（例如，总长度）。`parent[i]` 记录构成此最优链时，锚点 `i` 的直接前驱锚点的索引。
    *   **DP转移方程**：对于每个锚点 `i`，遍历所有在排序后位于其之前的锚点 `j`。如果锚点 `j` 可以连接到锚点 `i` (满足约束条件)，则尝试更新 `dp[i]`：
        `dp[i] = max(dp[i], dp[j] + score(anchor_i))`
        如果更新发生，则 `parent[i] = j`。
        `score(anchor_i)` 通常是 `kmersize`。
    *   这个DP过程在 `run.py` 的 `chain_anchors_kernel` Numba函数中实现，其复杂度为 O(n²)。
4.  **候选链提取**：通过回溯 `parent` 数组，从每个可能的结束锚点开始，可以重建得到多条候选的锚点链。过滤掉长度不足（锚点数少于 `min_anchors_param`）的链。
5.  **非重叠链选择（区间调度问题）**：从上一步得到的候选链集合中，选择一组互不重叠（在查询序列上）且总得分最高的链。这本身也是一个可以用动态规划解决的问题（加权区间调度）。
    *   将候选链按查询序列起始位置 `q_s` 排序。
    *   `dp_select[k]` 表示考虑前 `k` 条候选链时能得到的最大总分。
    *   这个过程在 `run.py` 的 `select_segments_kernel` Numba函数中实现。

### 细化伪代码 (主要针对 `chain_anchors_kernel` 核心逻辑)

```
输入：anchors_raw_data = [{q_s, r_s, strand, kmer_size}, ...], n = 锚点数
参数：max_gap_param, max_diag_diff_param, overlap_factor_param, min_anchors_param
输出：格式化的字符串，表示最终选定的非重叠链段

// === 步骤 0: 数据准备 (在主函数 function 中) ===
1. 从 anchors_raw_data 初始化 NumPy 数组: q_s_arr, q_e_arr, r_s_arr, r_e_arr, strand_arr
   (q_e = q_s + kmersize, r_e = r_s + kmersize)
2. 创建 py_anchors 列表 (字典形式)，用于后续处理和排序。
3. 按 (anchor['q_s'], anchor['r_s']) 升序排序 py_anchors。
4. 根据排序后的 py_anchors 更新 q_s_arr, q_e_arr, r_s_arr, r_e_arr, strand_arr，确保传递给 Numba 核函数的数据是排序后的。
5. 计算 max_allowed_overlap = int(kmersize * overlap_factor_param)

// === 步骤 1: 锚点链动态规划 (chain_anchors_kernel - Numba JIT) ===
// 输入: n_anchors (即 n), q_s_arr, q_e_arr, r_s_arr, r_e_arr, strand_arr (已排序)
//       kmersize, max_gap_between_anchors, max_diagonal_difference, max_allowed_overlap
Function chain_anchors_kernel:
  dp_score = array of size n_anchors, initialized to kmersize
  parent_idx = array of size n_anchors, initialized to -1

  For i from 0 to n_anchors-1: // 对于每个排序后的锚点 i
    anchor_i_q_s = q_s_arr[i]
    // ... (获取 anchor_i 的其他属性 q_e, r_s, r_e, strand)

    For j from 0 to i-1: // 对于锚点 i 之前的所有锚点 j
      anchor_j_q_s = q_s_arr[j]
      // ... (获取 anchor_j 的其他属性 q_e, r_s, r_e, strand)

      can_link = False
      If anchor_i_strand == anchor_j_strand:
        query_gap = anchor_i_q_s - anchor_j_q_e
        
        If anchor_i_strand == 1: // 正向链
          ref_gap = anchor_i_r_s - anchor_j_r_e
          diag_j = anchor_j_r_s - anchor_j_q_s
          diag_i = anchor_i_r_s - anchor_i_q_s
        Else: // 反向链 (strand == -1)
          ref_gap = anchor_j_r_s - anchor_i_r_e // 注意顺序
          diag_j = anchor_j_r_s + anchor_j_q_s
          diag_i = anchor_i_r_s + anchor_i_q_s
        
        If (-max_allowed_overlap <= query_gap <= max_gap_between_anchors) AND
           (-max_allowed_overlap <= ref_gap <= max_gap_between_anchors) AND
           (abs(diag_i - diag_j) <= max_diagonal_difference):
          can_link = True
      
      If can_link:
        current_chain_score = dp_score[j] + kmersize
        If current_chain_score > dp_score[i]:
          dp_score[i] = current_chain_score
          parent_idx[i] = j
  Return dp_score, parent_idx

// === 步骤 2: 形成候选链段 (在主函数 function 中) ===
1. For i from 0 to n_anchors-1:
     Reconstruct chain ending at anchor i using parent_idx.
     If chain_length >= min_anchors_param:
       Define segment_key (q_start, q_end, r_start, r_end) from chain.
       Store/update segment in candidate_segments_dict with its score (dp_score[i]).
2. Convert candidate_segments_dict.values() to candidate_segments list.
3. Filter candidate_segments by MIN_QUERY_LEN.
4. Sort filtered_segments by (seg['q_s'], -seg['score'], seg['q_e']).

// === 步骤 3: 选择非重叠链段 (select_segments_kernel - Numba JIT) ===
// 输入: n_segs, seg_q_s_arr, seg_q_e_arr, seg_scores_arr (已排序)
Function select_segments_kernel:
  dp_select_score = copy of seg_scores_arr
  prev_select_idx = array of size n_segs, initialized to -1

  For i from 0 to n_segs-1:
    seg_i_q_s = seg_q_s_arr[i]
    seg_i_score = seg_scores_arr[i]
    For j from 0 to i-1:
      seg_j_q_e = seg_q_e_arr[j]
      If seg_j_q_e <= seg_i_q_s: // Non-overlapping condition
        If dp_select_score[j] + seg_i_score > dp_select_score[i]:
          dp_select_score[i] = dp_select_score[j] + seg_i_score
          prev_select_idx[i] = j
  
  Find best_end_idx by maximizing dp_select_score.
  Reconstruct selected_indices path using prev_select_idx.
  Return selected_indices (in reverse order of selection)

// === 步骤 4: 输出格式化 (在主函数 function 中) ===
1. Reverse selected_indices to get correct order.
2. Construct final comma-separated string from the chosen segments.
```

#### 优化说明
-   **Numba JIT编译**：`chain_anchors_kernel` 和 `select_segments_kernel` 两个核心计算密集型函数使用Numba进行即时编译，将Python代码转换为高效的机器码，大幅提升运行速度。
-   **排序预处理**：锚点和候选链段在传入DP核心函数前都进行了排序，这是DP算法正确高效执行的前提。
-   **NumPy数组操作**：尽可能使用NumPy数组进行数据存储和传递，利用其底层C实现的效率。
-   **约束检查**：在`chain_anchors_kernel`中，严格的间隙、对角线和重叠检查确保了生物学意义上的合理连接。

---

## 三、时空复杂度分析

-   **锚点生成 (`seq2hashtable_multi_test` in `eval.py`)**:
    *   时间复杂度: 大致为 O(L_ref + L_query + N_matches)，其中 L_ref 和 L_query 是序列长度，N_matches 是初始k-mer匹配数。哈希表操作平均O(1)。
    *   空间复杂度: O(L_ref + N_matches) 用于存储哈希表和初始匹配。
-   **锚点排序 (Python `sort` in `run.py`)**:
    *   时间复杂度: O(N log N)，其中 N 是锚点总数。
    *   空间复杂度: O(N) 或 O(log N) 取决于具体排序算法实现，通常为O(N)用于存储排序后的数据或副本。
-   **锚点链式DP (`chain_anchors_kernel` in `run.py`)**:
    *   时间复杂度: O(N²)，因为存在两层嵌套循环遍历所有锚点对。
    *   空间复杂度: O(N) 用于存储 `dp_score` 和 `parent_idx` 数组。
-   **候选链提取和排序 (Python part in `run.py`)**:
    *   时间复杂度: 提取所有链最坏可达O(N²)，排序候选链O(M log M)，M为候选链数 (M <= N)。
    *   空间复杂度: O(M) 或 O(N) 存储候选链。
-   **非重叠链选择DP (`select_segments_kernel` in `run.py`)**:
    *   时间复杂度: O(M²)，M是候选链段的数量。
    *   空间复杂度: O(M) 用于存储DP状态。

**总体复杂度**:
由于 N (锚点数) 通常远大于 M (最终候选链段数)，算法的瓶颈在于 `chain_anchors_kernel`。
-   **时间复杂度**: O(L_ref + L_query + N log N + N²) ≈ **O(N²)** (假设锚点生成和排序时间被N²主导)。
-   **空间复杂度**: O(L_ref + N) ≈ **O(N)** (主要由锚点数据和DP数组决定)。
这满足实验要求的时间复杂度不高于平方量级。

---

## 四、代码实现说明

本项目包含以下主要Python脚本：

1.  **`run.py`**: 包含算法的核心逻辑。
    *   `function(...)`: 主协调函数，接收原始锚点数据和配置参数，执行完整的锚点链构建和筛选流程。它负责：
        *   数据预处理：将输入的原始锚点数据（通常是NumPy数组）转换为内部使用的数据结构，并进行排序。
        *   调用 `chain_anchors_kernel` 进行锚点链的动态规划计算。
        *   从 `chain_anchors_kernel` 的结果中提取候选的链段。
        *   调用 `select_segments_kernel` 从候选链段中选出最优的非重叠链段组合。
        *   格式化最终结果为字符串。
    *   `chain_anchors_kernel(...)`: 使用 Numba JIT 编译的函数。它实现了锚点链的动态规划算法。输入排序后的锚点信息，通过比较每对锚点是否满足连接约束（间隙、对角线差异、重叠），计算以每个锚点结尾的最长链得分。
    *   `select_segments_kernel(...)`: 使用 Numba JIT 编译的函数。它实现了基于加权区间调度的动态规划算法，用于从一组候选链段中选择查询序列上不重叠且总分最高的链段子集。

2.  **`eval.py`**: 包含评估和数据处理相关的函数。
    *   `seq2hashtable_multi_test(...)`: 用于从参照序列和查询序列中生成初始的k-mer锚点匹配。它构建参照序列的k-mer哈希表，然后扫描查询序列（及其反向互补序列）以快速找到匹配。
    *   `calculate_value(...)`: 根据算法输出的链段字符串和原始序列，计算对齐得分。它会解析链段，并使用 `edlib`库计算编辑距离，最终得出一个综合评分。
    *   `get_rc(...)`, `rc(...)`: 生成反向互补序列。
    *   `get_points(...)`, `format_tuples_for_display(...)`:辅助函数，用于解析和格式化链段字符串。

3.  **`data.py`**: 存储实验用的DNA序列数据（`ref1`, `que1`, `ref2`, `que2`）。

4.  **`tune_parameters.py`**: 用于自动化参数调优的脚本。
    *   定义了参数范围 (`PARAM_RANGES`)。
    *   使用 `itertools.product` 生成所有参数组合。
    *   利用 `multiprocessing.Pool` 并行地对每个参数组合调用 `evaluate_params` 函数。
    *   `evaluate_params(...)`: 对给定的参数组合，运行 `run.function` 并在两个数据集上使用 `eval.calculate_value` 评估得分。
    *   `update_and_save_results(...)`: 实时更新并保存当前找到的最佳参数组合到 `tuning_results_live.txt` 文件中，包括针对单个数据集的最优解和组合得分最优解。

代码结构清晰，通过Numba加速核心计算，并通过`tune_parameters.py`支持高效的参数搜索。

---

## 五、参数调优说明

为了获得最佳的序列比对结果，算法提供了多个可调参数。这些参数的设置对匹配的灵敏度和特异性有显著影响。`tune_parameters.py`脚本用于系统地搜索最优参数组合。

主要可调参数及其作用：

-   **`max_gap_param` (最大间隙)**: 定义在同一条链上，两个连续锚点之间在查询序列或参照序列上允许的最大距离（`query_gap` 或 `ref_gap`）。
    *   较小值：产生更连续、更紧凑的链，适用于序列相似度高、结构差异小的情况。
    *   较大值：允许链中存在较大的插入/删除或未匹配区域，适用于序列间差异较大或存在结构变异的情况。
-   **`max_diag_diff_param` (最大对角线差异)**: 限制两个连接锚点在对角线上的偏移量。对角线定义为 `r_s - q_s` (正向链) 或 `r_s + q_s` (反向链)。此参数控制了匹配的共线性程度。
    *   较小值：要求锚点严格共线，适用于没有大规模重排的序列。
    *   较大值：允许局部的小规模重排或重复序列导致的对角线跳跃。
-   **`overlap_factor_param` (重叠因子)**: 定义锚点间允许的最大重叠程度，以k-mer大小的百分比表示。实际允许的重叠像素数为 `int(kmersize * overlap_factor_param)`。重叠在这里被视为负的间隙。
    *   较小值 (如0.1-0.3)：允许少量重叠，可能有助于连接因测序错误或微小变异而略微错开的锚点。
    *   较大值 (如0.7-0.8)：允许显著重叠，可能在处理重复区域或复杂变异时有用，但需谨慎以防产生过多冗余匹配。
-   **`min_anchors_param` (最小锚点数)**: 一条候选链段必须包含的最小锚点数量。
    *   用于过滤掉由少量偶然匹配构成的短链，提高结果的可靠性。增加此值可以减少噪声，但可能丢失一些真实的短匹配区域。

**调参流程**:
`tune_parameters.py` 脚本采用网格搜索（Grid Search）的策略：
1.  为每个参数定义一个候选值列表。
2.  生成所有可能的参数组合。
3.  使用 `multiprocessing` 模块并行处理这些参数组合。
4.  对于每个组合，在预定义的两个数据集上运行完整的匹配算法 (`run.function`)，并使用 `eval.calculate_value` 计算得分。
5.  脚本会实时记录和更新在每个数据集上得分最高、以及组合得分最高的参数组合。
6.  结果会动态写入 `tuning_results_live.txt` 文件，方便监控调优进度和查看当前最优结果。

通过此过程，可以找到在特定数据集上表现最佳的参数配置。

---

## 六、运行结果

参数调优过程会生成详细的日志和结果。最终的调优结果，包括在不同数据集上表现最佳的参数组合及其对应的得分和比对详情，都记录在以下文件中：

详细的参数调优与运行结果请见：[tuning_results_live.txt](./tuning_results_live.txt)

**示例结果片段 (通常包含在 `tuning_results_live.txt` 中):**
```
--- Top 2 Results for Combined Score (Score1 + Score2) (So Far) ---
    1. Params: {'max_gap_param': 50, 'max_diag_diff_param': 20, 'overlap_factor_param': 0.1, 'min_anchors_param': 2}
    Score1: 29387, Alignment1: "0,2034,0,2034,2039,21605,8223,27802,21607,29824,21601,29825"
    Score2: 1830, Alignment2: "0,280,0,280,282,397,382,497,400,490,510,600,507,703,607,803,712,898,612,798,900,995,705,800,1000,1195,700,895,1200,1297,903,1000,1308,1391,908,991,1402,1495,402,495,1500,1596,1000,1096,1598,1693,1298,1393,1722,1768,1322,1368,1807,1899,1107,1199,2299,2498,1499,1698"
    Combined Score: 31217

    2. Params: {'max_gap_param': 50, 'max_diag_diff_param': 20, 'overlap_factor_param': 0.1, 'min_anchors_param': 1}
    Score1: 29387, Alignment1: "0,2034,0,2034,2039,21605,8223,27802,21607,29824,21601,29825"
    Score2: 1830, Alignment2: "0,280,0,280,282,397,382,497,400,490,510,600,507,703,607,803,712,898,612,798,900,995,705,800,1000,1195,700,895,1200,1297,903,1000,1308,1391,908,991,1402,1495,402,495,1500,1596,1000,1096,1598,1693,1298,1393,1722,1768,1322,1368,1807,1899,1107,1199,2299,2498,1499,1698"
    Combined Score: 31217
```
该文件会展示不同参数组合下的性能，帮助选择最优配置用于最终提交。

---

## 七、实验提交说明

为完成本实验，需提交以下内容：

1.  **GitHub仓库链接**：包含所有源代码（`.py`文件）、数据文件（如有必要，但本项目中数据嵌入在`data.py`）、以及本`README.md`实验报告。
2.  **评分网站提交**：使用通过参数调优得到的最佳参数配置，运行算法生成针对官方测试序列的匹配结果字符串，并将此字符串实名提交至指定的评分网站。
3.  **实验报告（本README.md文件）**：本文档作为实验报告，已详细描述了实验任务、算法设计、伪代码、时空复杂度、代码实现细节、参数调优方法和运行结果总结。

确保所有文件组织清晰，代码可运行，并且报告内容完整准确。

---
