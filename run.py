import numpy as np
from collections import defaultdict, deque
import numba
import itertools

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

# New brute force chain exploration for maximum scoring
@numba.njit(fastmath=True, cache=True)
def exhaustive_chain_kernel(n_anchors, q_s_arr, q_e_arr, r_s_arr, r_e_arr, strand_arr, 
                           kmersize, max_gap_between_anchors, max_diagonal_difference, max_allowed_overlap):
    """More aggressive chaining that explores all valid combinations"""
    # Initialize with single anchor scores
    best_scores = np.full(n_anchors, kmersize, dtype=np.int32)
    best_parents = np.full(n_anchors, -1, dtype=np.int32)
    
    if n_anchors <= 1:
        return best_scores, best_parents
    
    # Pre-compute diagonals
    diag_arr = np.empty(n_anchors, dtype=np.int32)
    for i in range(n_anchors):
        if strand_arr[i] == 1:
            diag_arr[i] = r_s_arr[i] - q_s_arr[i]
        else:
            diag_arr[i] = r_s_arr[i] + q_s_arr[i]
    
    # Multi-pass optimization for better chains
    for pass_num in range(3):  # Multiple passes for convergence
        updated = False
        
        for i in range(1, n_anchors):
            anchor_i_q_s = q_s_arr[i]
            anchor_i_r_s = r_s_arr[i]
            anchor_i_strand = strand_arr[i]
            anchor_i_diag = diag_arr[i]
            
            current_best = best_scores[i]
            current_parent = best_parents[i]
            
            # Try connecting to ALL previous anchors, not just immediate predecessors
            for j in range(i):
                if anchor_i_strand != strand_arr[j]:
                    continue
                
                # More lenient gap constraints for aggressive chaining
                query_gap = anchor_i_q_s - q_e_arr[j]
                if query_gap < -max_allowed_overlap * 2 or query_gap > max_gap_between_anchors * 1.5:
                    continue
                
                # Relaxed diagonal constraint
                if abs(anchor_i_diag - diag_arr[j]) > max_diagonal_difference * 1.2:
                    continue
                
                ref_gap = 0
                if anchor_i_strand == 1:
                    ref_gap = anchor_i_r_s - r_e_arr[j]
                else:
                    ref_gap = r_s_arr[j] - r_e_arr[i]
                
                if ref_gap < -max_allowed_overlap * 2 or ref_gap > max_gap_between_anchors * 1.5:
                    continue
                
                # Aggressive scoring with bonuses for longer chains
                base_score = best_scores[j] + kmersize
                
                # Bonus for close anchors (density bonus)
                if query_gap <= kmersize and ref_gap <= kmersize:
                    base_score += kmersize // 2
                
                # Bonus for diagonal consistency
                if abs(anchor_i_diag - diag_arr[j]) <= max_diagonal_difference // 2:
                    base_score += kmersize // 4
                
                if base_score > current_best:
                    current_best = base_score
                    current_parent = j
                    updated = True
            
            best_scores[i] = current_best
            best_parents[i] = current_parent
        
        if not updated:
            break
    
    return best_scores, best_parents

# Brute force segment selection for small datasets
def brute_force_segment_selection(candidate_segments, max_segments=12):
    """Try all possible combinations of non-overlapping segments"""
    if len(candidate_segments) <= max_segments:
        best_score = 0
        best_combination = []
        
        # Try all possible combinations
        for r in range(1, len(candidate_segments) + 1):
            for combination in itertools.combinations(range(len(candidate_segments)), r):
                # Check if segments in combination are non-overlapping
                valid = True
                total_score = 0
                
                for i in range(len(combination)):
                    seg_i = candidate_segments[combination[i]]
                    total_score += seg_i['score']
                    
                    for j in range(i + 1, len(combination)):
                        seg_j = candidate_segments[combination[j]]
                        # Check overlap
                        if not (seg_i['q_e'] <= seg_j['q_s'] or seg_j['q_e'] <= seg_i['q_s']):
                            valid = False
                            break
                    
                    if not valid:
                        break
                
                if valid and total_score > best_score:
                    best_score = total_score
                    best_combination = list(combination)
        
        return best_combination
    else:
        # Fall back to DP for large datasets
        return None

# Enhanced segment merging for overlapping segments
def aggressive_segment_merging(candidate_segments):
    """Merge overlapping segments aggressively to maximize score"""
    merged_segments = []
    
    # Sort by query start position
    segments = sorted(candidate_segments, key=lambda s: s['q_s'])
    
    i = 0
    while i < len(segments):
        current_seg = segments[i].copy()
        merged_count = 1
        
        # Look for overlapping segments to merge
        j = i + 1
        while j < len(segments):
            next_seg = segments[j]
            
            # Check for overlap or close proximity
            if (next_seg['q_s'] <= current_seg['q_e'] + 50 and  # Allow small gaps
                next_seg['strand'] == current_seg['strand']):
                
                # Merge segments
                current_seg['q_e'] = max(current_seg['q_e'], next_seg['q_e'])
                current_seg['r_s'] = min(current_seg['r_s'], next_seg['r_s'])
                current_seg['r_e'] = max(current_seg['r_e'], next_seg['r_e'])
                current_seg['score'] += next_seg['score']
                current_seg['length'] = current_seg['q_e'] - current_seg['q_s']
                merged_count += 1
                
                # Remove merged segment
                segments.pop(j)
            else:
                j += 1
        
        # Add bonus for merged segments
        if merged_count > 1:
            current_seg['score'] += merged_count * 10
        
        merged_segments.append(current_seg)
        i += 1
    
    return merged_segments

# Local neighborhood search for score optimization
def local_search_optimization(candidate_segments, initial_selection, search_radius=3):
    """Perform local search around the initial solution to find better scores"""
    if not initial_selection or not candidate_segments:
        return initial_selection
    
    best_selection = initial_selection[:]
    best_score = sum(candidate_segments[idx]['score'] for idx in initial_selection)
    
    # Get all possible segments that could be swapped
    all_indices = set(range(len(candidate_segments)))
    selected_indices = set(initial_selection)
    unselected_indices = all_indices - selected_indices
    
    improved = True
    iteration = 0
    max_iterations = 20
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Try removing segments and adding nearby ones
        for remove_idx in list(selected_indices):
            # Try removing this segment
            temp_selection = [idx for idx in best_selection if idx != remove_idx]
            temp_score = sum(candidate_segments[idx]['score'] for idx in temp_selection)
            
            # Find compatible segments to add
            for add_idx in unselected_indices:
                test_selection = temp_selection + [add_idx]
                
                # Check if selection is valid (non-overlapping)
                if is_valid_selection(candidate_segments, test_selection):
                    test_score = sum(candidate_segments[idx]['score'] for idx in test_selection)
                    
                    if test_score > best_score:
                        best_score = test_score
                        best_selection = test_selection[:]
                        selected_indices = set(best_selection)
                        unselected_indices = all_indices - selected_indices
                        improved = True
        
        # Try adding compatible segments without removing
        for add_idx in list(unselected_indices):
            test_selection = best_selection + [add_idx]
            
            if is_valid_selection(candidate_segments, test_selection):
                test_score = sum(candidate_segments[idx]['score'] for idx in test_selection)
                
                if test_score > best_score:
                    best_score = test_score
                    best_selection = test_selection[:]
                    selected_indices = set(best_selection)
                    unselected_indices = all_indices - selected_indices
                    improved = True
        
        # Try swapping segments
        for remove_idx in list(selected_indices):
            for add_idx in unselected_indices:
                test_selection = [idx for idx in best_selection if idx != remove_idx] + [add_idx]
                
                if is_valid_selection(candidate_segments, test_selection):
                    test_score = sum(candidate_segments[idx]['score'] for idx in test_selection)
                    
                    if test_score > best_score:
                        best_score = test_score
                        best_selection = test_selection[:]
                        selected_indices = set(best_selection)
                        unselected_indices = all_indices - selected_indices
                        improved = True
    
    return sorted(best_selection)

def is_valid_selection(candidate_segments, selection):
    """Check if a selection of segments is valid (non-overlapping)"""
    if not selection:
        return True
    
    segments = [candidate_segments[idx] for idx in selection]
    segments.sort(key=lambda s: s['q_s'])
    
    for i in range(len(segments) - 1):
        if segments[i]['q_e'] > segments[i + 1]['q_s']:
            return False
    
    return True

# Enhanced brute force with local optimization
def enhanced_brute_force_selection(candidate_segments, max_segments=15):
    """Enhanced brute force with local search optimization"""
    initial_selection = brute_force_segment_selection(candidate_segments, max_segments)
    
    if initial_selection is None:
        return None
    
    # Apply local search optimization
    optimized_selection = local_search_optimization(candidate_segments, initial_selection)
    
    return optimized_selection

# Greedy segment extension for score maximization
def greedy_segment_extension(candidate_segments):
    """Greedily extend segments by finding nearby anchors"""
    extended_segments = []
    
    for seg in candidate_segments:
        # Look for nearby segments that could be merged
        extended_seg = seg.copy()
        
        # Find segments that are close and could extend this one
        for other_seg in candidate_segments:
            if other_seg == seg or other_seg['strand'] != seg['strand']:
                continue
            
            # Check if segments are close enough to merge
            q_gap = abs(other_seg['q_s'] - seg['q_e'])
            r_gap = abs(other_seg['r_s'] - seg['r_e'])
            
            if q_gap <= 100 and r_gap <= 100:  # Close enough to merge
                # Extend the segment
                extended_seg['q_s'] = min(extended_seg['q_s'], other_seg['q_s'])
                extended_seg['q_e'] = max(extended_seg['q_e'], other_seg['q_e'])
                extended_seg['r_s'] = min(extended_seg['r_s'], other_seg['r_s'])
                extended_seg['r_e'] = max(extended_seg['r_e'], other_seg['r_e'])
                extended_seg['score'] += other_seg['score'] // 2  # Partial score bonus
                extended_seg['length'] = extended_seg['q_e'] - extended_seg['q_s']
        
        extended_segments.append(extended_seg)
    
    return extended_segments

# Multi-strategy segment generation
def multi_strategy_segments(py_anchors, dp_score, parent_idx, kmersize, min_anchors):
    """Generate segments using multiple strategies for maximum coverage"""
    all_segments = []
    
    # Strategy 1: Original chain-based segments
    for i in range(len(py_anchors)):
        chain_indices = []
        curr = i
        while curr != -1:
            chain_indices.append(curr)
            curr = parent_idx[curr]
        
        if len(chain_indices) >= min_anchors:
            chain_indices.reverse()
            first_idx, last_idx = chain_indices[0], chain_indices[-1]
            
            all_segments.append({
                'q_s': py_anchors[first_idx]['q_s'],
                'q_e': py_anchors[last_idx]['q_e'],
                'r_s': min(py_anchors[idx]['r_s'] for idx in chain_indices),
                'r_e': max(py_anchors[idx]['r_e'] for idx in chain_indices),
                'score': dp_score[i] + len(chain_indices) * kmersize // 4,
                'strand': py_anchors[i]['strand'],
                'length': py_anchors[last_idx]['q_e'] - py_anchors[first_idx]['q_s'],
                'chain_length': len(chain_indices),
                'strategy': 'chain'
            })
    
    # Strategy 2: Sliding window segments
    window_size = max(3, min_anchors)
    for i in range(len(py_anchors) - window_size + 1):
        window_anchors = py_anchors[i:i + window_size]
        
        # Check if all anchors in window have same strand
        if all(a['strand'] == window_anchors[0]['strand'] for a in window_anchors):
            all_segments.append({
                'q_s': window_anchors[0]['q_s'],
                'q_e': window_anchors[-1]['q_e'],
                'r_s': min(a['r_s'] for a in window_anchors),
                'r_e': max(a['r_e'] for a in window_anchors),
                'score': sum(kmersize for _ in window_anchors) + window_size * 5,
                'strand': window_anchors[0]['strand'],
                'length': window_anchors[-1]['q_e'] - window_anchors[0]['q_s'],
                'chain_length': window_size,
                'strategy': 'window'
            })
    
    # Strategy 3: Density-based segments
    for strand in [1, -1]:
        strand_anchors = [a for a in py_anchors if a['strand'] == strand]
        if len(strand_anchors) >= min_anchors:
            # Find dense regions
            for i in range(len(strand_anchors) - min_anchors + 1):
                for j in range(i + min_anchors, len(strand_anchors) + 1):
                    region_anchors = strand_anchors[i:j]
                    q_span = region_anchors[-1]['q_e'] - region_anchors[0]['q_s']
                    
                    # High density region
                    if q_span > 0 and len(region_anchors) / q_span > 0.01:
                        all_segments.append({
                            'q_s': region_anchors[0]['q_s'],
                            'q_e': region_anchors[-1]['q_e'],
                            'r_s': min(a['r_s'] for a in region_anchors),
                            'r_e': max(a['r_e'] for a in region_anchors),
                            'score': len(region_anchors) * kmersize + q_span // 5,
                            'strand': strand,
                            'length': q_span,
                            'chain_length': len(region_anchors),
                            'strategy': 'density'
                        })
    
    return all_segments

# Main aggressive alignment function
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
    
    # STAGE 3: Use exhaustive chaining for better results
    dp_score, parent_idx = exhaustive_chain_kernel(
        n_anchors, q_s_arr, q_e_arr, r_s_arr, r_e_arr, strand_arr,
        kmersize, MAX_GAP_BETWEEN_ANCHORS, MAX_DIAGONAL_DIFFERENCE, MAX_ALLOWED_OVERLAP
    )
    
    # STAGE 4: Form candidate segments with more aggressive criteria
    MIN_ANCHORS_PER_CHAIN = max(1, min_anchors_param // 2)  # More lenient
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
            
            # Enhanced scoring
            base_score = dp_score[i]
            length_bonus = len(chain_indices) * kmersize // 4
            coverage_bonus = (q_end - q_start) // 10
            
            candidate_segments.append({
                'q_s': q_start, 
                'q_e': q_end,
                'r_s': r_start, 
                'r_e': r_end,
                'score': base_score + length_bonus + coverage_bonus,
                'strand': py_anchors[i]['strand'],
                'length': q_end - q_start,
                'chain_length': len(chain_indices)
            })
    
    # If no valid segments, return empty string
    if not candidate_segments:
        return ""
    
    # STAGE 4.5: Aggressive segment merging
    candidate_segments = aggressive_segment_merging(candidate_segments)
    
    # STAGE 5: Try brute force selection for small datasets
    candidate_segments.sort(key=lambda s: (s['q_s'], -s['score'], s['q_e']))
    
    # Try brute force first
    selected_indices = brute_force_segment_selection(candidate_segments)
    
    if selected_indices is None:
        # Fall back to optimized DP
        n_segs = len(candidate_segments)
        seg_q_s_arr = np.array([s['q_s'] for s in candidate_segments], dtype=np.int32)
        seg_q_e_arr = np.array([s['q_e'] for s in candidate_segments], dtype=np.int32)
        seg_scores_arr = np.array([s['score'] for s in candidate_segments], dtype=np.int32)
        seg_lengths_arr = np.array([s['length'] for s in candidate_segments], dtype=np.int32)
        
        selected_indices_reversed = select_segments_kernel(
            n_segs, seg_q_s_arr, seg_q_e_arr, seg_scores_arr, seg_lengths_arr
        )
        
        if not selected_indices_reversed:
            return ""
        
        selected_indices = list(reversed(selected_indices_reversed))
    
    # STAGE 6: Format output with selected segments
    selected_segments = [candidate_segments[idx] for idx in selected_indices]
    selected_segments.sort(key=lambda s: s['q_s'])  # Ensure proper order
    
    output_parts = []
    for seg in selected_segments:
        output_parts.extend([str(seg['q_s']), str(seg['q_e']), str(seg['r_s']), str(seg['r_e'])])
    
    return ",".join(output_parts)

# Provide a simplified version for backward compatibility
def simple_function(data):
    return function(data)
