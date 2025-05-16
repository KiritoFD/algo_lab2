# Improved DNA Sequence Alignment Algorithm

## Core Enhancements

1. **Adaptive K-mer Selection**
   - Use variable k-mer sizes based on sequence complexity
   - Shorter k-mers (7-9) for diversity-rich regions
   - Longer k-mers (13-15) for highly conserved regions
   - Combine seeds from multiple k-mer lengths for better coverage

2. **Hierarchical Chaining with Gap-Aware Scoring**
   - Implement a more sophisticated scoring function:
     - Higher rewards for longer exact matches
     - Penalties proportional to gap size
     - Lower penalties for gaps with similar sequence composition
   - First chain high-confidence seeds, then fill gaps with smaller seeds

3. **Adaptive Segment Extension**
   - After identifying core aligned segments, extend boundaries using:
     - Semi-global alignment at segment edges
     - Allow small mismatches if overall similarity remains high

4. **Experiment-Specific Optimizations**
   - For Experiment 1 (large contiguous alignment):
     - Prioritize long, continuous segments
     - More aggressive merging of nearby segments
   - For Experiment 2 (fragmented alignment):
     - Better handling of short, high-quality segments
     - Dynamic threshold adjustment based on segment density

5. **Ensemble Approach**
   - Generate multiple candidate alignments with different parameters
   - Score each candidate with the evaluation function
   - Select the highest-scoring result
