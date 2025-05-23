import numpy as np
from collections import defaultdict, deque
import numba

# Optimized anchor chaining with early filtering and improved handling of strands
@numba.njit(fastmath=True, cache=True)
def chain_anchors_kernel(n_anchors, q_s_arr, q_e_arr, r_s_arr, r_e_arr, strand_arr, 
                         kmersize, max_gap_between_anchors, max_diagonal_difference, max_allowed_overlap):
    # Pre-allocate arrays for dynamic programming
    dp_score = np.full(n_anchors, kmersize, dtype=np.int32)
    parent_idx = np.full(n_anchors, -1, dtype=np.int32)

    # Early exit for empty input
    if n_anchors <= 1:
        return dp_score, parent_idx
        
    # Pre-compute anchor diagonals to avoid redundant calculations
    diag_arr = np.empty(n_anchors, dtype=np.int32)
    for i in range(n_anchors):
        if strand_arr[i] == 1:  # Forward strand
            diag_arr[i] = r_s_arr[i] - q_s_arr[i]
        else:  # Reverse strand
            diag_arr[i] = r_s_arr[i] + q_s_arr[i]
    
    # Main dynamic programming loop
    for i in range(1, n_anchors):  # Start from second anchor
        anchor_i_q_s = q_s_arr[i]
        anchor_i_q_e = q_e_arr[i]
        anchor_i_r_s = r_s_arr[i]
        anchor_i_r_e = r_e_arr[i]
        anchor_i_strand = strand_arr[i]
        anchor_i_diag = diag_arr[i]
        
        # Variable to track the best chain score found for anchor i
        best_chain_score = dp_score[i]
        best_parent = -1

        # Binary search could be used here for large datasets to find a reasonable j_start
        # For now, we scan all previous anchors as the datasets are relatively small
        for j in range(i):
            # Skip if different strands - quick filter
            if anchor_i_strand != strand_arr[j]:
                continue
                
            anchor_j_q_e = q_e_arr[j]
            anchor_j_r_e = r_e_arr[j]
            
            # Check query gap first (most common reject)
            query_gap = anchor_i_q_s - anchor_j_q_e
            if query_gap < -max_allowed_overlap or query_gap > max_gap_between_anchors:
                continue
            
            # Check diagonal difference (next most common reject)
            if abs(anchor_i_diag - diag_arr[j]) > max_diagonal_difference:
                continue
                
            # Check reference gap (strand-specific)
            ref_gap = 0
            if anchor_i_strand == 1:  # Forward strand
                ref_gap = anchor_i_r_s - anchor_j_r_e
            else:  # Reverse strand
                ref_gap = r_s_arr[j] - anchor_i_r_e
                
            if ref_gap < -max_allowed_overlap or ref_gap > max_gap_between_anchors:
                continue
            
            # If we get here, anchors can be linked
            current_chain_score = dp_score[j] + kmersize
            if current_chain_score > best_chain_score:
                best_chain_score = current_chain_score
                best_parent = j
        
        # Update with the best parent found
        if best_parent != -1:
            dp_score[i] = best_chain_score
            parent_idx[i] = best_parent
            
    return dp_score, parent_idx

# Optimized segment selection with improved scoring and linear space
@numba.njit(fastmath=True, cache=True)
def select_segments_kernel(n_segs, seg_q_s_arr, seg_q_e_arr, seg_scores_arr, seg_lengths_arr=None):
    # Improved scoring that considers segment length
    if seg_lengths_arr is None:
        seg_lengths_arr = seg_q_e_arr - seg_q_s_arr
        
    # Initial dynamic programming arrays
    dp_select_score = np.copy(seg_scores_arr)
    prev_select_idx = np.full(n_segs, -1, dtype=np.int32)

    # Bottom-up dynamic programming
    for i in range(n_segs):
        seg_i_q_s = seg_q_s_arr[i]
        seg_i_score = seg_scores_arr[i]
        
        # Binary search for the largest j where seg_j_q_e <= seg_i_q_s would be more efficient
        # for very large datasets, but linear search is simpler and works well for moderate sizes
        for j in range(i):
            seg_j_q_e = seg_q_e_arr[j]
            if seg_j_q_e <= seg_i_q_s:  # Non-overlapping segments
                if dp_select_score[j] + seg_i_score > dp_select_score[i]:
                    dp_select_score[i] = dp_select_score[j] + seg_i_score
                    prev_select_idx[i] = j
    
    # Find segment with maximum score
    best_end_idx = -1
    if n_segs > 0:
        best_end_idx = np.argmax(dp_select_score)
    
    # Reconstruct the optimal path using backtracking
    selected_indices = []
    curr_idx = best_end_idx
    while curr_idx != -1:
        selected_indices.append(curr_idx)
        curr_idx = prev_select_idx[curr_idx]
    
    # Numba doesn't support list.reverse(), will be reversed in Python
    return selected_indices

# Main alignment function with optimized workflow
def function(data, 
             max_gap_param=250, 
             max_diag_diff_param=150, 
             overlap_factor_param=0.5, 
             min_anchors_param=1):
    """
    Perform sequence alignment using k-mer anchors.
    
    Args:
        data: NumPy array with each row as (query_pos, ref_pos, strand, kmersize)
        max_gap_param: Maximum allowed gap between anchors (default: 250)
        max_diag_diff_param: Maximum allowed diagonal difference (default: 150)
        overlap_factor_param: Factor for calculating allowed anchor overlap (default: 0.5)
        min_anchors_param: Minimum number of anchors required per valid chain (default: 1)
        
    Returns:
        String representing alignment coordinates as "q_s,q_e,r_s,r_e,q_s,q_e,r_s,r_e,..."
        Empty string if no valid alignment is found.
    """
    # Validate input data
    if not isinstance(data, np.ndarray) or data.size == 0:
        return ""

    # Handle singleton array
    if data.ndim == 1 and len(data) >= 4:
        data = np.array([data])
    
    if data.shape[0] == 0:
        return ""

    # Extract K-mer size
    kmersize = int(data[0, 3])
    if kmersize <= 0:
        return ""
        
    # STAGE 1: Initialize data structures
    n_anchors = data.shape[0]
    
    # Create typed arrays for Numba
    q_s_arr = np.empty(n_anchors, dtype=np.int32)
    q_e_arr = np.empty(n_anchors, dtype=np.int32)
    r_s_arr = np.empty(n_anchors, dtype=np.int32)
    r_e_arr = np.empty(n_anchors, dtype=np.int32)
    strand_arr = np.empty(n_anchors, dtype=np.int8)
    
    # Python data structures for reconstruction
    py_anchors = []
    
    # Fill arrays with anchor data
    for i in range(n_anchors):
        q_s, r_s, strand_val = int(data[i, 0]), int(data[i, 1]), int(data[i, 2])
        q_s_arr[i] = q_s
        q_e_arr[i] = q_s + kmersize
        r_s_arr[i] = r_s
        r_e_arr[i] = r_s + kmersize
        strand_arr[i] = strand_val
        py_anchors.append({
            'q_s': q_s, 
            'q_e': q_s + kmersize, 
            'r_s': r_s, 
            'r_e': r_s + kmersize, 
            'strand': strand_val, 
            'id': i
        })
    
    # STAGE 2: Sort anchors and prepare arrays
    # Sort by query position first, then reference position
    py_anchors.sort(key=lambda a: (a['q_s'], a['r_s']))
    
    # Update sorted arrays
    for i in range(n_anchors):
        anchor = py_anchors[i]
        q_s_arr[i] = anchor['q_s']
        q_e_arr[i] = anchor['q_e']
        r_s_arr[i] = anchor['r_s']
        r_e_arr[i] = anchor['r_e']
        strand_arr[i] = anchor['strand']
    
    # STAGE 3: Chain anchors using dynamic programming
    MAX_GAP_BETWEEN_ANCHORS = max_gap_param
    MAX_DIAGONAL_DIFFERENCE = max_diag_diff_param
    MAX_ALLOWED_OVERLAP = int(kmersize * overlap_factor_param)
    
    dp_score, parent_idx = chain_anchors_kernel(
        n_anchors, q_s_arr, q_e_arr, r_s_arr, r_e_arr, strand_arr,
        kmersize, MAX_GAP_BETWEEN_ANCHORS, MAX_DIAGONAL_DIFFERENCE, MAX_ALLOWED_OVERLAP
    )
    
    # STAGE 4: Form candidate segments from chains
    MIN_ANCHORS_PER_CHAIN = min_anchors_param
    candidate_segments = []
    
    # For each anchor, reconstruct the chain ending at that anchor
    for i in range(n_anchors):
        # Build chain by backtracking
        chain_indices = []
        curr = i
        while curr != -1:
            chain_indices.append(curr)
            curr = parent_idx[curr]
        
        # Proceed if chain meets minimum length requirement
        if len(chain_indices) >= MIN_ANCHORS_PER_CHAIN:
            # Reverse to get the chain in forward order
            chain_indices.reverse()
            
            # Extract chain endpoints
            first_idx = chain_indices[0]
            last_idx = chain_indices[-1]
            
            # Create segment from chain
            q_start = py_anchors[first_idx]['q_s']
            q_end = py_anchors[last_idx]['q_e']
            
            # Find min r_start and max r_end within the chain
            r_starts = [py_anchors[idx]['r_s'] for idx in chain_indices]
            r_ends = [py_anchors[idx]['r_e'] for idx in chain_indices]
            r_start = min(r_starts) if r_starts else 0
            r_end = max(r_ends) if r_ends else 0
            
            # Add segment to candidates
            candidate_segments.append({
                'q_s': q_start, 
                'q_e': q_end,
                'r_s': r_start, 
                'r_e': r_end,
                'score': dp_score[i],  # Score from dynamic programming
                'strand': py_anchors[i]['strand'],
                'length': q_end - q_start  # Store length for potential filtering
            })
    
    # If no valid segments, return empty string
    if not candidate_segments:
        return ""
    
    # STAGE 5: Select optimal non-overlapping segments
    # Sort segments by query start position, then by score (decreasing), then by length
    candidate_segments.sort(key=lambda s: (s['q_s'], -s['score'], s['q_e']))
    
    n_segs = len(candidate_segments)
    seg_q_s_arr = np.array([s['q_s'] for s in candidate_segments], dtype=np.int32)
    seg_q_e_arr = np.array([s['q_e'] for s in candidate_segments], dtype=np.int32)
    seg_scores_arr = np.array([s['score'] for s in candidate_segments], dtype=np.int32)
    seg_lengths_arr = np.array([s['length'] for s in candidate_segments], dtype=np.int32)
    
    # Get indices of selected segments (in reverse order)
    selected_indices_reversed = select_segments_kernel(
        n_segs, seg_q_s_arr, seg_q_e_arr, seg_scores_arr, seg_lengths_arr
    )
    
    # No valid selection found
    if not selected_indices_reversed:
        return ""
    
    # STAGE 6: Format output
    # Reverse indices to get correct order
    selected_segments = []
    for idx in reversed(selected_indices_reversed):
        selected_segments.append(candidate_segments[idx])
    
    # Build output string
    output_parts = []
    for seg in selected_segments:
        output_parts.extend([str(seg['q_s']), str(seg['q_e']), str(seg['r_s']), str(seg['r_e'])])
    
    return ",".join(output_parts)

# Provide a simplified version for backward compatibility
def simple_function(data):
    return function(data)
