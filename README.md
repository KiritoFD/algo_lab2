# DNA Sequence Alignment Project

## Project Overview

This project implements a highly optimized DNA sequence alignment system that uses k-mer anchoring and dynamic programming techniques to efficiently align query sequences against reference sequences. The system supports both forward and reverse-complement strand alignment, with customizable parameters to optimize performance for different types of sequence data.

## Project Structure

The project consists of several key components:

- **Core Algorithm Implementation:**
  - `run.py` - Python implementation with Numba JIT acceleration
  - `run.cpp` - C++ implementation for production use

- **Evaluation Framework:**
  - `eval.cpp` - C++ code for testing and benchmarking alignments

- **Parameter Optimization Tools:**
  - `parameter_tuning.cpp` - Two-stage simulated annealing optimizer
  - `gradient_optimizer.cpp` - Adaptive gradient ascent optimizer

- **Documentation:**
  - `algorithm_design.md` - Detailed algorithm description
  - `README.md` - This project overview

## Algorithm Description

### Core Approach

The alignment algorithm uses a multi-stage approach:

1. **K-mer Matching:** Identify exact k-mer matches (anchors) between query and reference sequences
2. **Anchor Chaining:** Use dynamic programming to form chains of co-linear anchors
3. **Segment Formation:** Convert chains into alignment segments
4. **Non-overlapping Selection:** Select an optimal set of non-overlapping segments

### Key Components

#### K-mer Matching (in eval.cpp)
- Uses a hash table to efficiently find matching k-mers
- Supports both forward and reverse complement strands
- Implemented in `seq2hashtable_multi_test()`

#### Anchor Chaining (in run.py/run.cpp)
- Dynamic programming to find optimal chains of anchors
- Considers diagonal consistency, gap constraints, and strand orientation
- Python version uses Numba JIT compilation for speed

#### Segment Selection (in run.py/run.cpp)
- Another dynamic programming step to select non-overlapping segments
- Maximizes total alignment coverage and score
- Handles trade-offs between coverage and alignment quality

## Parameters

The algorithm's behavior can be fine-tuned through several parameters:

- `max_gap_param`: Maximum allowed gap between anchors (default: 250)
- `max_diag_diff_param`: Maximum allowed diagonal difference (default: 150)
- `overlap_factor_param`: Factor for calculating allowed anchor overlap (default: 0.5)
- `min_anchors_param`: Minimum number of anchors required per valid chain (default: 1)

## Parameter Optimization

The project includes two parameter optimization approaches:

### Simulated Annealing (parameter_tuning.cpp)
- Two-stage approach that first optimizes for Dataset1, then Dataset2
- Uses random jumps to escape local optima
- Performs cross-evaluation between datasets

### Adaptive Gradient Ascent (gradient_optimizer.cpp)
- Per-parameter adaptive step sizes
- Patience mechanism to avoid premature step size reduction
- Random jumps after extended periods of no improvement

## Performance Considerations

- The Python implementation uses Numba JIT compilation for near-C performance
- Pre-computation of diagonals and early filtering improve efficiency
- The C++ implementation provides maximum performance for production use
- Both implementations use dynamic programming with careful memory management

## Compilation and Usage

### Compilation

```bash
# Compile the evaluation framework and main algorithm
make eval

# Compile parameter tuning tools
make parameter_tuning
make gradient_optimizer
```

### Basic Usage

```bash
# Run the evaluation on test datasets
./eval

# Run parameter tuning
./parameter_tuning

# Run gradient optimization
./gradient_optimizer
```

### Input Format

The input data consists of:
- Reference sequence files (e.g., ref1.txt, ref2.txt)
- Query sequence files (e.g., que1.txt, que2.txt)

### Output Format

The alignment results are returned as comma-separated values in the format:
```
q_start1,q_end1,r_start1,r_end1,q_start2,q_end2,r_start2,r_end2,...
```
Where:
- `q_start`, `q_end`: Query sequence alignment start and end positions
- `r_start`, `r_end`: Reference sequence alignment start and end positions

## Algorithm Enhancements

The optimized version in `run.py` includes several enhancements:
1. Pre-computation of anchor diagonals
2. Early filtering of incompatible anchors
3. Improved strand handling
4. Optimized segment selection with segment length consideration
5. Strategic use of Numba's `fastmath` and `cache` options

## Extension Points

The modular design allows for several extension points:
- Alternative anchor chaining algorithms
- Different segment selection strategies
- Custom scoring functions
- Hardware-specific optimizations (e.g., SIMD, GPU)
