import numpy as np
from collections import defaultdict, deque
import numba

# Numba JIT compiled function for anchor chaining
@numba.njit
def chain_anchors_kernel(n_anchors, q_s_arr, q_e_arr, r_s_arr, r_e_arr, strand_arr, 
                         kmersize, max_gap_between_anchors, max_diagonal_difference, max_allowed_overlap): # Parameters passed
    dp_score = np.full(n_anchors, kmersize, dtype=np.int32)
    parent_idx = np.full(n_anchors, -1, dtype=np.int32)

    for i in range(n_anchors):
        anchor_i_q_s = q_s_arr[i]
        anchor_i_q_e = q_e_arr[i]
        anchor_i_r_s = r_s_arr[i]
        anchor_i_r_e = r_e_arr[i]
        anchor_i_strand = strand_arr[i]

        for j in range(i):
            anchor_j_q_s = q_s_arr[j]
            anchor_j_q_e = q_e_arr[j]
            anchor_j_r_s = r_s_arr[j]
            anchor_j_r_e = r_e_arr[j]
            anchor_j_strand = strand_arr[j]
            
            can_link = False
            if anchor_i_strand == anchor_j_strand:
                query_gap = anchor_i_q_s - anchor_j_q_e
                
                diag_j, diag_i = 0, 0
                ref_gap = 0

                if anchor_i_strand == 1: # Forward strand
                    ref_gap = anchor_i_r_s - anchor_j_r_e
                    diag_j = anchor_j_r_s - anchor_j_q_s
                    diag_i = anchor_i_r_s - anchor_i_q_s
                else: # Reverse strand (strand == -1)
                    ref_gap = anchor_j_r_s - anchor_i_r_e
                    diag_j = anchor_j_r_s + anchor_j_q_s
                    diag_i = anchor_i_r_s + anchor_i_q_s
                
                if (-max_allowed_overlap <= query_gap <= max_gap_between_anchors and # Use parameter
                    -max_allowed_overlap <= ref_gap <= max_gap_between_anchors): # Use parameter
                    if abs(diag_i - diag_j) <= max_diagonal_difference: # Use parameter
                        can_link = True
            
            if can_link:
                current_chain_score = dp_score[j] + kmersize
                if current_chain_score > dp_score[i]:
                    dp_score[i] = current_chain_score
                    parent_idx[i] = j
    return dp_score, parent_idx

# Numba JIT compiled function for selecting non-overlapping segments
@numba.njit
def select_segments_kernel(n_segs, seg_q_s_arr, seg_q_e_arr, seg_scores_arr):
    dp_select_score = np.copy(seg_scores_arr) # Ensure it's a copy if seg_scores_arr is from a slice
    prev_select_idx = np.full(n_segs, -1, dtype=np.int32)

    for i in range(n_segs):
        seg_i_q_s = seg_q_s_arr[i]
        seg_i_score = seg_scores_arr[i] # Use original score for addition
        
        for j in range(i):
            seg_j_q_e = seg_q_e_arr[j]
            if seg_j_q_e <= seg_i_q_s: # Non-overlapping
                if dp_select_score[j] + seg_i_score > dp_select_score[i]:
                    dp_select_score[i] = dp_select_score[j] + seg_i_score
                    prev_select_idx[i] = j
    
    best_total_score = 0
    best_end_idx = -1
    if n_segs > 0:
        current_max_score = np.int32(-1) # Numba needs explicit type for initial min
        for i in range(n_segs):
            if dp_select_score[i] > current_max_score:
                current_max_score = dp_select_score[i]
                best_end_idx = i
        if best_end_idx != -1: # Check if any valid score was found
             best_total_score = current_max_score


    # Reconstruct path
    selected_indices = []
    if best_end_idx != -1:
        curr_idx = best_end_idx
        while curr_idx != -1:
            selected_indices.append(curr_idx)
            curr_idx = prev_select_idx[curr_idx]
    # Numba list.reverse() is not supported, so we append and then reverse in Python if needed,
    # or build it in reverse order if that's acceptable.
    # For now, returning as is, Python side can reverse.
    return selected_indices # This will be in reverse order of selection


def function(data, 
             max_gap_param=250, 
             max_diag_diff_param=150, 
             overlap_factor_param=0.5, 
             min_anchors_param=1): # Changed default min_anchors_param to 1
    if not isinstance(data, np.ndarray) or data.size == 0:
        return ""

    if data.ndim == 1: 
        if data.shape[0] < 4: 
             return ""
        data = np.array([data]) 

    if data.shape[0] == 0:
        return ""

    kmersize = int(data[0, 3])
    if kmersize <= 0: 
        return ""

    # 1. Anchor Representation & Parameters
    # Convert to NumPy arrays for Numba
    n_anchors = data.shape[0]
    q_s_arr = np.empty(n_anchors, dtype=np.int32)
    q_e_arr = np.empty(n_anchors, dtype=np.int32)
    r_s_arr = np.empty(n_anchors, dtype=np.int32)
    r_e_arr = np.empty(n_anchors, dtype=np.int32)
    strand_arr = np.empty(n_anchors, dtype=np.int8)
    
    # Original anchors list for segment reconstruction (easier with dicts there)
    py_anchors = [] 

    for i in range(n_anchors):
        q_s, r_s, strand_val = int(data[i, 0]), int(data[i, 1]), int(data[i, 2])
        q_s_arr[i] = q_s
        q_e_arr[i] = q_s + kmersize
        r_s_arr[i] = r_s
        r_e_arr[i] = r_s + kmersize
        strand_arr[i] = strand_val
        py_anchors.append({'q_s': q_s, 'q_e': q_s + kmersize, 'r_s': r_s, 'r_e': r_s + kmersize, 'strand': strand_val, 'id': i})


    # 2. Sorting Anchors (Sort indices based on q_s, r_s)
    # Numba works on contiguous arrays. Sorting data directly or using sorted indices.
    # For simplicity, let's sort the Python list of dicts and then rebuild arrays if needed,
    # or pass sorted indices to Numba.
    # Current Numba kernel expects data to be pre-sorted based on how 'j' iterates up to 'i'.
    # So, we sort the original data and then create the arrays.
    
    # Create a combined array for sorting, then extract
    # This is a bit inefficient but ensures data passed to Numba is sorted as expected.
    # A more optimal way would be to sort indices and then gather.
    
    # Let's sort py_anchors first, then create the numba arrays from the sorted py_anchors
    py_anchors.sort(key=lambda a: (a['q_s'], a['r_s']))

    for i in range(n_anchors):
        anchor = py_anchors[i]
        q_s_arr[i] = anchor['q_s']
        q_e_arr[i] = anchor['q_e']
        r_s_arr[i] = anchor['r_s']
        r_e_arr[i] = anchor['r_e']
        strand_arr[i] = anchor['strand']
        # py_anchors[i]['id'] = i # Update id to be the sorted index

    if n_anchors == 0: # Should be caught by data.shape[0] == 0 earlier
        return ""

    # 3. Chaining Anchors - Use passed parameters
    MAX_GAP_BETWEEN_ANCHORS = max_gap_param    
    MAX_DIAGONAL_DIFFERENCE = max_diag_diff_param      
    MAX_ALLOWED_OVERLAP = int(kmersize * overlap_factor_param) # Calculate from factor

    dp_score, parent_idx = chain_anchors_kernel(n_anchors, q_s_arr, q_e_arr, r_s_arr, r_e_arr, strand_arr,
                                                kmersize, MAX_GAP_BETWEEN_ANCHORS, MAX_DIAGONAL_DIFFERENCE, MAX_ALLOWED_OVERLAP)
    
    # 4. Forming Candidate Segments from Chains (Python part)
    # Modified to be more "brute-force": do not de-duplicate segments by coordinates here.
    MIN_ANCHORS_PER_CHAIN = min_anchors_param 
    candidate_segments_list = [] # Changed from candidate_segments_dict

    for i in range(n_anchors):
        current_chain_indices_py = []
        curr = i
        num_anchors_in_chain = 0
        while curr != -1:
            current_chain_indices_py.append(curr) # These are indices into the *sorted* py_anchors
            num_anchors_in_chain +=1
            curr = parent_idx[curr]
        current_chain_indices_py.reverse()

        if num_anchors_in_chain >= MIN_ANCHORS_PER_CHAIN:
            first_anchor_sorted_idx = current_chain_indices_py[0]
            last_anchor_sorted_idx = current_chain_indices_py[-1]

            q_start = py_anchors[first_anchor_sorted_idx]['q_s']
            q_end = py_anchors[last_anchor_sorted_idx]['q_e']
            
            chain_r_starts = [py_anchors[idx]['r_s'] for idx in current_chain_indices_py]
            chain_r_ends = [py_anchors[idx]['r_e'] for idx in current_chain_indices_py]
            
            r_start = min(chain_r_starts) if chain_r_starts else 0 
            r_end = max(chain_r_ends) if chain_r_ends else 0 
            
            current_segment_score = dp_score[i] 

            # Append all segments derived from valid chains
            candidate_segments_list.append({
                'q_s': q_start, 'q_e': q_end,
                'r_s': r_start, 'r_e': r_end,
                'score': current_segment_score, 
                'strand': py_anchors[i]['strand'] 
            })
    
    # candidate_segments = list(candidate_segments_dict.values()) # No longer needed
    candidate_segments = candidate_segments_list # Use the list directly

    # 5. Filtering and Selecting Non-Overlapping Segments
    # MIN_QUERY_LEN = 30 # Removed this filter to be more "brute-force"
    # filtered_segments_py = [seg for seg in candidate_segments if (seg['q_e'] - seg['q_s']) >= MIN_QUERY_LEN]
    filtered_segments_py = candidate_segments # Pass all candidate segments

    if not filtered_segments_py:
        return ""

    filtered_segments_py.sort(key=lambda s: (s['q_s'], -s['score'], s['q_e']))

    n_segs = len(filtered_segments_py)
    if n_segs == 0:
        return ""
        
    seg_q_s_arr = np.array([s['q_s'] for s in filtered_segments_py], dtype=np.int32)
    seg_q_e_arr = np.array([s['q_e'] for s in filtered_segments_py], dtype=np.int32)
    seg_scores_arr = np.array([s['score'] for s in filtered_segments_py], dtype=np.int32) # Use chain score

    selected_indices_reversed = select_segments_kernel(n_segs, seg_q_s_arr, seg_q_e_arr, seg_scores_arr)
    
    final_selected_segments = []
    # Numba returns list in reverse order of finding them by backtracking
    for i in range(len(selected_indices_reversed) -1, -1, -1): # Iterate to reverse
        final_selected_segments.append(filtered_segments_py[selected_indices_reversed[i]])
    
    # 6. Output Formatting
    output_parts = []
    for seg in final_selected_segments:
        output_parts.extend([str(seg['q_s']), str(seg['q_e']), str(seg['r_s']), str(seg['r_e'])])
    
    return ",".join(output_parts)
