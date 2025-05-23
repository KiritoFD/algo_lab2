# Enhanced Sequence Alignment Algorithm Documentation

## 1. Introduction

This document provides a comprehensive explanation of the enhanced sequence alignment algorithm implemented in `run.py`. The algorithm efficiently aligns a query sequence to a reference sequence using k-mer anchors, which are exact matches between the two sequences. These anchors are then chained together and selected to form optimal alignments.

The algorithm uses dynamic programming at two key stages: anchor chaining and non-overlapping segment selection. The implementation is optimized using Numba JIT compilation for high performance, making it suitable for processing genomic sequences with tens of thousands of k-mer matches.

## 2. Algorithm Overview

The sequence alignment process follows these major steps:

1. **Input Processing**: Convert k-mer matches into anchor structures
2. **Anchor Sorting**: Sort anchors by their positions to enable efficient processing
3. **Anchor Chaining**: Chain compatible anchors using dynamic programming
4. **Segment Formation**: Form candidate segments from anchor chains
5. **Segment Selection**: Select optimal non-overlapping segments
6. **Output Formatting**: Format the selected segments as coordinates

The following sections provide detailed explanations of each step.

## 3. Algorithm Details

### 3.1 Input Format and Preprocessing

**Input Data**: A NumPy array where each row represents a k-mer match with the format:
```
(query_position, reference_position, strand, kmer_size)
```

- `query_position` and `reference_position`: Starting positions of the k-mer in the query and reference
- `strand`: 1 for forward strand matches, -1 for reverse strand matches
- `kmer_size`: Length of the k-mer

**Anchor Representation**:
Each anchor is expanded to include the ending positions:
```
{
  'q_s': query_start,
  'q_e': query_end,
  'r_s': reference_start,
  'r_e': reference_end,
  'strand': strand,
  'id': original_index
}
```

### 3.2 Anchor Sorting

The anchors are sorted by:
1. Query start position (`q_s`) as the primary key
2. Reference start position (`r_s`) as the secondary key

This sorting is crucial for the efficiency of the dynamic programming algorithms that follow, as it ensures that potential predecessors of an anchor are processed before the anchor itself.

### 3.3 Anchor Chaining with Dynamic Programming

The anchor chaining step identifies maximal-scoring chains of co-linear anchors using dynamic programming.

**Mathematical Formulation**:

Let $DP[i]$ be the maximum score of a chain ending with anchor $i$.

Base case: $DP[i] = kmersize$ for all $i$ (each anchor initially scores its own length)

Recurrence relation:
For each anchor $i = 1, 2, ..., n$:
- For each previous anchor $j < i$:
  - If anchor $j$ can be linked to anchor $i$:
    - $DP[i] = max(DP[i], DP[j] + kmersize)$
    - If updated, set parent[i] = j

**Linking Criteria**:
Two anchors $j$ and $i$ (where $j$ comes before $i$ in the sorted order) can be linked if:
1. They are on the same strand
2. The query gap satisfies: $-MAX\_ALLOWED\_OVERLAP \leq query\_gap \leq MAX\_GAP\_BETWEEN\_ANCHORS$
3. The reference gap satisfies: $-MAX\_ALLOWED\_OVERLAP \leq ref\_gap \leq MAX\_GAP\_BETWEEN\_ANCHORS$
4. The diagonal difference is within limits: $|diag_i - diag_j| \leq MAX\_DIAGONAL\_DIFFERENCE$

Where:
- For forward strand: $diag = r\_s - q\_s$
- For reverse strand: $diag = r\_s + q\_s$
- $query\_gap = q\_s_i - q\_e_j$
- $ref\_gap$ depends on strand:
  - Forward: $ref\_gap = r\_s_i - r\_e_j$
  - Reverse: $ref\_gap = r\_s_j - r\_e_i$

**Optimization Techniques**:
1. Pre-compute diagonals to avoid redundant calculations
2. Quick-reject checks to avoid unnecessary computation:
   - Check strand match first
   - Then check query gap, which is most likely to fail
   - Then check diagonal difference
   - Finally check reference gap
3. Use Numba's fastmath and caching for improved performance

**Time Complexity**: O(n²) where n is the number of anchors
**Space Complexity**: O(n) for the dynamic programming arrays

### 3.4 Forming Candidate Segments

After the dynamic programming step, each anchor has a score and a possible parent anchor. For each anchor, we:

1. Reconstruct the chain by following the parent pointers
2. Filter out chains with fewer than `MIN_ANCHORS_PER_CHAIN` anchors
3. For each qualifying chain, create a candidate segment:
   - Query start = start of first anchor in chain
   - Query end = end of last anchor in chain
   - Reference start = minimum reference start among all anchors in chain
   - Reference end = maximum reference end among all anchors in chain
   - Score = score from dynamic programming (sum of k-mer sizes in chain)
   - Strand = strand of the anchors in the chain

### 3.5 Selecting Non-Overlapping Segments

The final step is to select an optimal set of non-overlapping segments using another dynamic programming approach.

**Mathematical Formulation**:

Let $DP[i]$ be the maximum score achievable by selecting non-overlapping segments ending with segment $i$.

Base case: $DP[i] = score[i]$ for all $i$

Recurrence relation:
For each segment $i = 0, 1, ..., n-1$:
- For each previous segment $j < i$:
  - If segment $j$ does not overlap with segment $i$ (i.e., $q\_e_j \leq q\_s_i$):
    - $DP[i] = max(DP[i], DP[j] + score[i])$
    - If updated, set prev[i] = j

After filling the DP table, the optimal solution is constructed by:
1. Finding the segment $i$ with the maximum $DP[i]$ value
2. Backtracking using the prev array to reconstruct the full set of selected segments

**Optimization Techniques**:
1. Pre-sorting segments by query start position
2. Using Numba JIT compilation for performance
3. Considering segment lengths in the scoring function (optional)

**Time Complexity**: O(m²) where m is the number of candidate segments
**Space Complexity**: O(m) for the dynamic programming arrays

### 3.6 Output Formatting

The final selected segments are formatted as a comma-separated string containing query and reference coordinates:

```
q_start_1,q_end_1,r_start_1,r_end_1,q_start_2,q_end_2,r_start_2,r_end_2,...
```

## 4. Algorithm Parameters

The algorithm's behavior is controlled by several key parameters:

1. **max_gap_param** (default: 250): Maximum allowed gap between consecutive anchors in both query and reference sequences.
2. **max_diag_diff_param** (default: 150): Maximum allowed difference in diagonal values for anchors to be considered co-linear.
3. **overlap_factor_param** (default: 0.5): Factor used to calculate allowed anchor overlap; MAX_ALLOWED_OVERLAP = kmersize * overlap_factor_param.
4. **min_anchors_param** (default: 1): Minimum number of anchors required to form a valid chain.

## 5. Performance Analysis

### 5.1 Time Complexity

- **Overall**: O(n² + m²)
  - n = number of input anchors
  - m = number of candidate segments (typically m << n)

- **Bottlenecks**:
  - Anchor chaining: O(n²)
  - Segment selection: O(m²)

### 5.2 Space Complexity

- **Overall**: O(n + m)
  - O(n) for anchor data structures
  - O(m) for segment data structures

### 5.3 Optimizations

- **Numba JIT compilation**: Speeds up inner loops by 10-100x
- **Early rejections**: Reduces unnecessary computations
- **Binary search**: Could be added for large datasets to find valid segment predecessors more quickly
- **Vectorization**: Some operations are vectorized with NumPy

## 6. Visualization and Examples

### Example Workflow

1. **Input**: K-mer matches between query and reference sequences
```
[(10, 100, 1, 15), (25, 115, 1, 15), (40, 130, 1, 15), ...]
```

2. **Output**: Selected alignment segments
```
10,55,100,145,70,120,200,250,...
```

### Visual Representation

```
Query:   --[===]-----[======]---[====]------
Reference: ---[===]---[======]-----[====]---

Legend:
[===]: Selected alignment segments
-----: Unaligned regions
```

## 7. Implementation Notes

### 7.1 Code Structure

- `chain_anchors_kernel`: JIT-compiled function for anchor chaining
- `select_segments_kernel`: JIT-compiled function for segment selection
- `function`: Main entry point that orchestrates the alignment process

### 7.2 Future Improvements

- **Parallelization**: Process different chains in parallel
- **Sparse Dynamic Programming**: For very large datasets
- **Adaptive Parameter Selection**: Automatically adjust parameters based on input characteristics
- **Quality Scoring**: Incorporate base quality scores for better alignments

## 8. Conclusion

The enhanced sequence alignment algorithm provides an efficient and accurate way to align query sequences to reference sequences using k-mer anchors. Its optimization for performance makes it suitable for genomic applications involving long sequences and many k-mer matches.

The algorithm's strength lies in its ability to handle both forward and reverse strand matches, and to select an optimal set of non-overlapping alignment segments that maximize the total alignment score.
