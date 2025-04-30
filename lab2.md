# 复杂DNA序列比对算法文档

---

## 一、问题描述
给定一对DNA参考序列（`ref`）和查询序列（`query`），目标是找出`query`中每个子片段与`ref`中哪一部分匹配，并输出形如`(q_start, q_end, r_start, r_end)`的元组列表，表示`query[q_start:q_end]`与`ref[r_start:r_end]`匹配。

支持以下变异类型：
- 单核苷酸突变（SNP）
- 插入/删除（Indel）
- 倒位（Reverse Complement）
- 片段移位（Fragment Shift）

要求：时间复杂度低于 $ O(mn) $

---

## 二、总体思路

采用**锚点哈希 + 图建模 + 最长路径搜索**策略，具体步骤如下：

1. **K-mer哈希生成候选锚点**
2. **图建模：锚点作为节点，边表示相邻匹配关系**
3. **动态规划寻找最长路径**
4. **局部精调优化边界**
5. **输出匹配结果**

---

## 三、算法流程详解

### Step 1: K-mer哈希与候选锚点生成

#### 功能
将`ref`和`query`划分为固定长度k的子串（k-mer），通过哈希表快速查找潜在匹配位置，合并连续重叠区域形成候选锚点。

#### 参数建议
- `k = 10~15`
- `min_anchor_len = 20`
- `max_freq = 10`（过滤高频k-mer避免重复干扰）

#### 实现伪代码
```python
def generate_kmers(seq, k):
    return [(seq[i:i+k], i) for i in range(len(seq)-k+1)]

def build_hash_table(ref_kmers):
    hash_table = defaultdict(list)
    for kmer, pos in ref_kmers:
        hash_table[kmer].append(pos)
    return hash_table

def find_candidate_anchors(query_kmers, hash_table, min_len=20, max_freq=10):
    candidates = []
    for kmer, q_pos in query_kmers:
        if kmer in hash_table and len(hash_table[kmer]) <= max_freq:
            for r_pos in hash_table[kmer]:
                candidates.append((q_pos, r_pos))

    # 合并连续k-mer为锚点
    anchors = []
    current = None
    for q_pos, r_pos in sorted(candidates, key=lambda x: (x[0], x[1])):
        if not current:
            current = [q_pos, q_pos + k, r_pos, r_pos + k]
        else:
            if q_pos == current[1] and r_pos == current[3]:
                current[1] += 1
                current[3] += 1
            else:
                if current[1] - current[0] >= min_len:
                    anchors.append(tuple(current))
                current = [q_pos, q_pos + k, r_pos, r_pos + k]
    if current and current[1] - current[0] >= min_len:
        anchors.append(tuple(current))
    return anchors
```

---

### Step 2: 图建模与权重分配

#### 功能
将锚点建模为图节点，建立合法转移边，计算权重用于路径选择。

#### 节点定义
四元组 `(q_start, q_end, r_start, r_end)`

#### 边建立规则
若锚点A在B之前且无重叠，则建立有向边 A → B。

#### 权重设计
$$ w(A,B) = \text{score}(B) - \lambda \cdot \text{gap_penalty}(A,B) $$

其中：
- `score(B)`：匹配长度（q_end - q_start）
- `gap_penalty`：插入/删除代价（距离之和 × λ）

#### 实现伪代码
```python
def build_graph(anchors, lambda_gap=2):
    G = nx.DiGraph()
    G.add_node("S")
    G.add_node("T")
    anchors.sort(key=lambda x: (x[2], x[0]))  # 按照r_start排序

    for i, anchor in enumerate(anchors):
        q_start, q_end, r_start, r_end = anchor
        score = q_end - q_start
        G.add_node(anchor, score=score)
        G.add_edge("S", anchor, weight=score)

        for j in range(i+1, len(anchors)):
            next_anchor = anchors[j]
            qr_start_ok = next_anchor[0] > q_end
            rf_start_ok = next_anchor[2] > r_end
            if qr_start_ok and rf_start_ok:
                gap = abs(next_anchor[0] - q_end) + abs(next_anchor[2] - r_end)
                edge_weight = (next_anchor[1] - next_anchor[0]) - lambda_gap * gap
                G.add_edge(anchor, next_anchor, weight=edge_weight)

    for anchor in anchors:
        G.add_edge(anchor, "T", weight=0)
    return G
```

---

### Step 3: 最长路径搜索（动态规划）

#### 功能
使用拓扑排序保证处理顺序，动态规划更新最优得分，最终回溯得到最佳锚点链。

#### 实现伪代码
```python
def longest_path_dp(G):
    topo_order = list(nx.topological_sort(G))
    dp = {node: 0 for node in topo_order}
    prev = {node: None for node in topo_order}

    for u in topo_order:
        for v in G.successors(u):
            new_score = dp[u] + G.edges[u][v]['weight']
            if new_score > dp[v]:
                dp[v] = new_score
                prev[v] = u

    path = []
    curr = "T"
    while curr != "S":
        path.append(curr)
        curr = prev[curr]
    path.reverse()
    return path
```

---

### Step 4: 局部精调与结果输出

#### 功能
对非完全匹配区域进行Smith-Waterman或Needleman-Wunsch比对修正边界，组合结果输出。

#### 实现伪代码
```python
def refine_alignment(path, ref, query):
    refined = []
    for i in range(1, len(path)-1):  # 排除虚拟起点终点
        curr = path[i]
        next_node = path[i+1] if i+1 < len(path) else None

        if next_node:
            q_region = query[curr[1]:next_node[0]]
            r_region = ref[curr[3]:next_node[2]]
            alignment = custom_local_align(q_region, r_region)
            refined.extend(alignment)
        else:
            refined.append(curr)
    return refined

def output_format(refined_anchors):
    result = []
    for anchor in refined_anchors:
        if anchor not in ("S", "T"):
            q_start, q_end, r_start, r_end = anchor
            result.append((q_start, q_end, r_start, r_end))
    return sorted(result, key=lambda x: x[0])
```

---

## 四、倒位与片段移位处理

### 倒位检测

增加反向互补序列匹配逻辑：
```python
def reverse_complement(seq):
    return ''.join(['T' if c == 'A' else 'A' if c == 'T' else 
                    'C' if c == 'G' else 'G' if c == 'C' else c for c in reversed(seq)])

# 在generate_kmers时同时处理正向和反向互补
```

### 片段移位检测

允许不连续锚点连接：
```python
# 修改build_graph中的条件判断
if next_anchor[2] > r_end or next_anchor[0] > q_end:  # 允许跨区域连接
```

---

## 五、完整主函数流程

```python
def main_pipeline(ref, query, k=15):
    # Step 1: 生成锚点
    ref_kmers = generate_kmers(ref, k)
    query_kmers = generate_kmers(query, k)
    hash_table = build_hash_table(ref_kmers)
    anchors = find_candidate_anchors(query_kmers, hash_table)

    # Step 2: 构建图模型
    G = build_graph(anchors)

    # Step 3: 动态规划找最长路径
    best_path = longest_path_dp(G)

    # Step 4: 局部精调与输出
    refined = refine_alignment(best_path, ref, query)
    return output_format(refined)
```

---

## 六、时空复杂度分析

| 步骤 | 时间复杂度 | 空间复杂度 |
|------|------------|------------|
| K-mer哈希 | $O(m + n)$ | $O(m)$ |
| 锚点生成 | $O(n + a)$ | $O(a)$ |
| 图构建 | $O(a^2)$（稀疏图优化后接近$O(a \log a)$） | $O(a + e)$ |
| 最长路径搜索 | $O(a \log a)$ | $O(a)$ |
| 局部精调 | $O(a \cdot l)$ | $O(l)$ |
| **总计** | $O((m + n) + a \log a + a \cdot l)$ | $O(m + a + e)$ |

- $m, n$：reference/query长度
- $a$：锚点数量（远小于$m,n$）
- $e$：图边数
- $l$：局部区域平均长度

---

## 七、示例输入输出

### 输入
```python
ref = "ATGGTACGA---TTC"
query = "ATG---CGAGTATTC"
```

### 输出
```python
[(0, 3, 0, 3),     # 匹配 ATG → ATG
 (3, 6, 6, 9),     # 匹配 CGA → CGA（跨越删除）
 (6, 10, 9, 13)]   # 匹配 TATTC → TTC（插入突变）
```

---

## 八、扩展方向

- **GPU加速**：使用CUDA进行哈希查询与动态规划优化。
- **多序列联合比对**：通过一致性锚点集拼接多个query。
- **错误容忍机制**：允许一定错配率，提升鲁棒性。
- **自适应k值选择**：根据GC含量调整k，提升哈希效率。

---

## 九、评分与提交说明

- 提交地址：
  - 第一组测试：http://10.20.26.11:8550
  - 第二组测试：http://10.20.26.11:8551
- 截止时间：2025年5月31日 23:59
- 提交内容：
  - Elearning实验报告（包含算法伪代码、复杂度分析、运行结果）
  - GitHub项目链接
  - 实名提交至评分网站

---

该算法兼顾效率与准确性，适用于真实生物数据分析场景，具备良好的可扩展性和工程实现性。