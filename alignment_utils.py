"""
Advanced DNA Sequence Alignment Utilities
"""

from collections import defaultdict
import bisect

def build_optimized_kmer_index(ref: str, k: int = 15, skip_n: bool = True) -> dict:
    """
    Builds an optimized k-mer index from reference sequence.
    
    Args:
        ref: Reference sequence
        k: k-mer size
        skip_n: Whether to skip k-mers containing N
        
    Returns:
        Dictionary mapping k-mers to their positions in reference
    """
    index = defaultdict(list)
    n_kmer = 'N' * k
    
    for i in range(len(ref) - k + 1):
        kmer = ref[i:i+k]
        # Skip k-mers with N if specified
        if skip_n and 'N' in kmer:
            continue
        # Store position
        index[kmer].append(i)
    
    return index

def find_optimized_seeds(query: str, ref_index: dict, k: int = 15) -> list:
    """
    Finds seeds (exact k-mer matches) between query and reference efficiently.
    
    Args:
        query: Query sequence
        ref_index: Reference k-mer index from build_optimized_kmer_index
        k: k-mer size
        
    Returns:
        List of (q_pos, r_pos) tuples
    """
    seeds = []
    
    for i in range(len(query) - k + 1):
        kmer = query[i:i+k]
        # Skip k-mers with N
        if 'N' in kmer:
            continue
        # Find matches in reference
        for r_pos in ref_index.get(kmer, []):
            seeds.append((i, r_pos))
    
    return seeds

def chain_seeds_optimized(seeds: list, max_gap: int = 50) -> list:
    """
    Chains seeds using patience sorting algorithm with O(n log n) complexity.
    
    Args:
        seeds: List of (q_pos, r_pos) seed matches
        max_gap: Maximum allowed gap between consecutive seeds
        
    Returns:
        List of seeds in optimal chain
    """
    if not seeds:
        return []
    
    # Sort seeds by query position and reference position
    seeds.sort(key=lambda s: (s[0], s[1]))
    
    # Initialize data structures for patience sorting
    piles = []  # Last r_pos in each pile
    backlinks = []  # Index of preceding seed in chain
    pile_indices = []  # Index of seed at top of each pile
    
    for i, (q_pos, r_pos) in enumerate(seeds):
        # Find the pile where this seed belongs using binary search
        pile_idx = bisect.bisect_left([p[-1][1] for p in piles] if piles else [], r_pos)
        
        # Create new backlink
        if pile_idx > 0:
            backlinks.append(pile_indices[pile_idx - 1])
        else:
            backlinks.append(-1)
        
        # Update piles
        if pile_idx == len(piles):
            piles.append([(q_pos, r_pos)])
            pile_indices.append(i)
        else:
            piles[pile_idx].append((q_pos, r_pos))
            pile_indices[pile_idx] = i
    
    # Reconstruct the chain
    chain = []
    if piles:
        idx = pile_indices[-1]
        while idx >= 0:
            chain.append(seeds[idx])
            idx = backlinks[idx]
        chain.reverse()
    
    return chain

def merge_and_filter_segments(chain: list, ref: str, query: str, k: int,
                            max_edit_ratio: float = 0.05, min_length: int = 30) -> list:
    """
    Merges chained seeds into segments and filters by quality.
    
    Args:
        chain: Chain of seeds from chain_seeds_optimized
        ref: Reference sequence
        query: Query sequence
        k: k-mer size
        max_edit_ratio: Maximum allowed edit distance ratio
        min_length: Minimum segment length
        
    Returns:
        List of (q_start, q_end, r_start, r_end) segments
    """
    from evaluate import calculate_distance
    
    if not chain:
        return []
    
    segments = []
    current_q_start, current_r_start = chain[0]
    prev_q_pos, prev_r_pos = chain[0]
    
    for q_pos, r_pos in chain[1:]:
        # Check if continuous
        if q_pos == prev_q_pos + 1 and r_pos == prev_r_pos + 1:
            prev_q_pos, prev_r_pos = q_pos, r_pos
            continue
        
        # End current segment and check quality
        q_end = prev_q_pos + k
        r_end = prev_r_pos + k
        
        if q_end - current_q_start >= min_length:
            edit_dist = calculate_distance(
                ref, query, current_r_start, r_end, 
                current_q_start, q_end
            )
            
            if edit_dist / (q_end - current_q_start) <= max_edit_ratio:
                segments.append((current_q_start, q_end, current_r_start, r_end))
        
        # Start new segment
        current_q_start, current_r_start = q_pos, r_pos
        prev_q_pos, prev_r_pos = q_pos, r_pos
    
    # Process the last segment
    q_end = prev_q_pos + k
    r_end = prev_r_pos + k
    
    if q_end - current_q_start >= min_length:
        edit_dist = calculate_distance(
            ref, query, current_r_start, r_end, 
            current_q_start, q_end
        )
        
        if edit_dist / (q_end - current_q_start) <= max_edit_ratio:
            segments.append((current_q_start, q_end, current_r_start, r_end))
    
    return segments

def align_sequence_optimized(query: str, ref: str, k: int = 13, 
                         max_gap: int = 50, max_edit_ratio: float = 0.05) -> list:
    """
    Complete DNA sequence alignment pipeline with optimized algorithms.
    
    Args:
        query: Query sequence
        ref: Reference sequence
        k: k-mer size
        max_gap: Maximum gap for chaining
        max_edit_ratio: Maximum edit distance ratio
        
    Returns:
        List of (q_start, q_end, r_start, r_end) alignment segments
    """
    # Build k-mer index from reference
    ref_index = build_optimized_kmer_index(ref, k)
    
    # Find seeds (exact k-mer matches)
    seeds = find_optimized_seeds(query, ref_index, k)
    
    # Chain seeds
    chain = chain_seeds_optimized(seeds, max_gap)
    
    # Merge into segments and filter by quality
    segments = merge_and_filter_segments(chain, ref, query, k, max_edit_ratio)
    
    return segments
