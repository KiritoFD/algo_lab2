# Lab2: 复杂DNA序列比对算法

## 项目概述

本项目实现了基于锚点链接的复杂DNA序列比对算法，能够处理多种DNA变异类型，包括单核苷酸突变(SNV)、插入、删除、重复、倒位和片段移位等。

## 算法设计

### 核心思想：基于锚点的链式比对

算法采用三阶段流水线处理：

1. **锚点发现（Anchor Finding）**：从输入数据中提取k-mer精确匹配作为锚点
2. **锚点链接（Anchor Chaining）**：使用动态规划将兼容的锚点连接成链
3. **片段选择（Segment Selection）**：选择非重叠的最优片段组合

### 算法流程

```
1. 将输入数据转换为锚点结构体
2. 按query和reference位置对锚点排序
3. 使用动态规划链接锚点，约束条件包括：
   - 相同链方向要求
   - 最大间隔约束
   - 对角线差异约束
4. 从锚点链生成候选片段
5. 选择非重叠的最高分片段组合
6. 输出比对片段的标准格式
```

### 关键特性

- **时间复杂度**：O(k²) 其中k为锚点数量，远优于传统O(mn)方法
- **空间复杂度**：O(k) 线性空间复杂度
- **处理多种变异**：支持SNV、插入删除、倒位、片段移位等
- **基于图算法**：将比对问题建模为有向无环图中的最长路径问题

### 算法参数

- `max_gap_param`：锚点间最大允许间隔（默认：250）
- `max_diag_diff_param`：锚点链接的最大对角线差异（默认：150）
- `overlap_factor_param`：重叠容忍因子（默认：0.5）
- `min_anchors_param`：每个链的最少锚点数（默认：1）

## 输入输出格式

### 输入
- 元组向量：`(query_start, ref_start, strand, kmer_size)`
- 每个元组表示一个k-mer匹配锚点

### 输出
- 比对片段的字符串表示
- 格式：`(query_start, query_end, ref_start, ref_end)`

## 核心数据结构

### 锚点结构体
```cpp
struct Anchor {
    int q_s, q_e;      // Query序列起始/结束位置
    int r_s, r_e;      // Reference序列起始/结束位置
    int strand;        // 链方向信息（1：正向，-1：反向）
    int id;           // 锚点标识符
    int score;        // 锚点得分
};
```

### 片段结构体
```cpp
struct Segment {
    int q_s, q_e;      // Query序列起始/结束位置
    int r_s, r_e;      // Reference序列起始/结束位置
    int score;        // 片段得分
    int strand;       // 链方向信息
};
```

## 核心算法函数

### 1. 锚点链接算法
```cpp
std::pair<std::vector<int>, std::vector<int>> chain_anchors(
    const std::vector<Anchor>& anchors,
    int kmersize,
    int max_gap_between_anchors,
    int max_diagonal_difference,
    int max_allowed_overlap
);
```

**功能**：使用动态规划将兼容的锚点连接成链

**约束条件**：
- 相同链方向（正向或反向）
- Query和Reference间隔在允许范围内
- 对角线差异在阈值内

### 2. 片段选择算法
```cpp
std::vector<int> select_segments(const std::vector<Segment>& segments);
```

**功能**：从候选片段中选择非重叠的最优组合

**策略**：动态规划求解最大权重独立集问题

### 3. 主函数
```cpp
std::string function(
    const std::vector<std::tuple<int, int, int, int>>& data,
    int max_gap_param = 250,
    int max_diag_diff_param = 150,
    double overlap_factor_param = 0.5,
    int min_anchors_param = 1
);
```

**功能**：协调整个比对流程的主入口函数

## 复杂度分析

### 时间复杂度
- **锚点链接**：O(k²) 其中k为锚点数量
- **片段选择**：O(s²) 其中s为片段数量
- **总体复杂度**：O(k²) 通常 k << mn，远优于传统方法

### 空间复杂度
- **锚点存储**：O(k)
- **动态规划数组**：O(k)
- **总体复杂度**：O(k) 线性空间

## 算法优势

1. **高效性**：避免了传统O(mn)的完整动态规划矩阵计算
2. **准确性**：基于生物学约束的锚点链接保证比对质量
3. **灵活性**：可调参数适应不同类型的DNA变异
4. **扩展性**：图算法框架便于后续功能扩展

## 使用方法

```cpp
#include "run.cpp"

// 准备输入数据
std::vector<std::tuple<int, int, int, int>> data;
// ... 填充数据 ...

// 调用比对函数
std::string result = function(data, 250, 150, 0.5, 1);
```

## 测试环境

算法适用于指定的评估系统：
- 第一组数据集：http://10.20.26.11:8550
- 第二组数据集：http://10.20.26.11:8551

## 文件结构

```
algo_lab2/
├── README.md          # 项目说明文档
├── run.cpp           # 主要算法实现
└── eval.cpp          # 评估接口（如果提供）
```

## 实验要求达成

- ✅ **图算法应用**：将比对建模为图中最长路径问题
- ✅ **复杂度要求**：O(k²) < O(mn) 满足小于平方级别要求
- ✅ **变异处理**：支持多种DNA变异类型
- ✅ **输出格式**：符合指定的元组格式要求

## 参考文献

- 基于锚点的序列比对算法
- 生物信息学中的动态规划方法
- 图算法在生物信息学中的应用
