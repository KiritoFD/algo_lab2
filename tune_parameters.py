import itertools
import multiprocessing
import numpy as np
import os
import time
from heapq import nlargest, heappushpop, heappush

# Assuming run.py, eval.py, data.py are in the same directory or accessible via PYTHONPATH
from run import function as run_function # The refactored function from run.py
from eval import seq2hashtable_multi_test, calculate_value # Functions from eval.py
from data import ref1, que1, ref2, que2 # Data from data.py

# Define expanded parameter ranges to explore
PARAM_RANGES = {
    "max_gap_param": list(range(50, 401, 5)) + [450, 500],  # More steps, wider range
    # Example: [50, 75, 100, ..., 400, 450, 500]
    "max_diag_diff_param": list(range(20, 201, 10)) + [250, 300, 350], # More steps, wider range
    # Example: [20, 40, ..., 200, 250, 300, 350]
    "overlap_factor_param": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], # More fine-grained
    "min_anchors_param": [1, 2, 3, 4, 5] # Slightly expanded
}

# Global cache for k-mer data
kmer_data_cache = {}

# Globals for tracking best results dynamically in the main process
# These will store the full result tuple: (params, s1, t1, s2, t2, cs)
best_overall_for_score1_so_far = None
best_overall_for_score2_so_far = None

N_TOP = 2
# Using min-heaps to store the N largest scores. Store (-score, result_tuple) for min-heap to find N largest.
# Or, more simply, just keep appending and re-sorting with nlargest for small N_TOP.
# Let's use lists and nlargest for simplicity in update logic.
top_n_for_score1_list = []
top_n_for_score2_list = []
top_n_for_combined_list = []

# Lock for writing to the output file if we decide to make saving more frequent from callback
# For now, saving will be sequential in the main loop.
output_lock = multiprocessing.Lock() 
output_filename = "tuning_results_live.txt"


# Global counter for progress (for worker processes to update)
progress_counter_mp = None
total_combinations_global_mp = 0

def init_worker(counter, total_combs_shared):
    global progress_counter_mp, total_combinations_global_mp
    progress_counter_mp = counter
    total_combinations_global_mp = total_combs_shared

def get_kmer_data(dataset_id, ref, query, kmersize=9, shift=1):
    if dataset_id not in kmer_data_cache:
        print(f"Generating k-mer data for dataset {dataset_id}...")
        kmer_data_cache[dataset_id] = seq2hashtable_multi_test(ref, query, kmersize=kmersize, shift=shift)
        print(f"Dataset {dataset_id} k-mer matches shape: {kmer_data_cache[dataset_id].shape}")
    return kmer_data_cache[dataset_id]

def evaluate_params(params_tuple):
    # This function remains largely the same, calculates scores and returns them.
    # Progress printing is moved to the main loop after result is received.
    max_gap, max_diag_diff, overlap_factor, min_anchors = params_tuple
    current_params = {
        "max_gap_param": max_gap,
        "max_diag_diff_param": max_diag_diff,
        "overlap_factor_param": overlap_factor,
        "min_anchors_param": min_anchors
    }
    # pid = os.getpid() # Not needed for console output from worker anymore

    data1 = get_kmer_data(1, ref1, que1)
    score1 = 0
    tuples_str1 = ""
    if data1.size > 0:
        tuples_str1 = str(run_function(data1, **current_params))
        score1 = calculate_value(tuples_str1, ref1, que1)
    
    data2 = get_kmer_data(2, ref2, que2)
    score2 = 0
    tuples_str2 = ""
    if data2.size > 0:
        tuples_str2 = str(run_function(data2, **current_params))
        score2 = calculate_value(tuples_str2, ref2, que2)
        
    if progress_counter_mp is not None: # Update shared counter from worker
        with progress_counter_mp.get_lock():
            progress_counter_mp.value += 1
            
    return current_params, score1, tuples_str1, score2, tuples_str2, score1 + score2


def update_and_save_results(new_result, total_combinations_count):
    global best_overall_for_score1_so_far, best_overall_for_score2_so_far
    global top_n_for_score1_list, top_n_for_score2_list, top_n_for_combined_list
    
    params, s1, t1, s2, t2, cs = new_result

    # Update overall best for score1
    if best_overall_for_score1_so_far is None or s1 > best_overall_for_score1_so_far[1]:
        best_overall_for_score1_so_far = new_result
    
    # Update overall best for score2
    if best_overall_for_score2_so_far is None or s2 > best_overall_for_score2_so_far[3]:
        best_overall_for_score2_so_far = new_result

    # Update top N lists (append and then use nlargest)
    # This is less efficient than a heap for large N or many updates, but simpler for N=2
    top_n_for_score1_list.append(new_result)
    top_n_for_score1_list = nlargest(N_TOP, top_n_for_score1_list, key=lambda x: x[1])

    top_n_for_score2_list.append(new_result)
    top_n_for_score2_list = nlargest(N_TOP, top_n_for_score2_list, key=lambda x: x[3])

    top_n_for_combined_list.append(new_result)
    top_n_for_combined_list = nlargest(N_TOP, top_n_for_combined_list, key=lambda x: x[5])

    # Write current bests to file
    with output_lock: # Ensure file writing is atomic if this were called from multiple threads/processes
        try:
            with open(output_filename, "w") as f:
                f.write(f"Parameter Tuning Results (Live Update)\n")
                f.write(f"Tested approximately {progress_counter_mp.value if progress_counter_mp else 0}/{total_combinations_count} combinations.\n\n")

                f.write("--- Overall Best for Dataset 1 Score (So Far) ---\n")
                if best_overall_for_score1_so_far:
                    p, s, t, _, _, _ = best_overall_for_score1_so_far # Unpack only relevant parts for this section
                    f.write(f"Params: {p}\n")
                    f.write(f"   Score1: {s}, Alignment1: \"{t}\"\n\n")
                else:
                    f.write("Awaiting results...\n\n")

                f.write("--- Overall Best for Dataset 2 Score (So Far) ---\n")
                if best_overall_for_score2_so_far:
                    p, _, _, s, t, _ = best_overall_for_score2_so_far
                    f.write(f"Params: {p}\n")
                    f.write(f"   Score2: {s}, Alignment2: \"{t}\"\n\n")
                else:
                    f.write("Awaiting results...\n\n")
                
                f.write(f"--- Top {N_TOP} Results for Dataset 1 Score (So Far) ---\n")
                for i, res_tuple in enumerate(top_n_for_score1_list):
                    p, s_1, t_1, s_2, t_2, c_s = res_tuple
                    f.write(f"{i+1}. Params: {p}\n")
                    f.write(f"   Score1: {s_1}, Alignment1: \"{t_1}\"\n")
                    f.write(f"   Score2: {s_2}, Alignment2: \"{t_2}\"\n")
                    f.write(f"   Combined Score: {c_s}\n\n")

                f.write(f"--- Top {N_TOP} Results for Dataset 2 Score (So Far) ---\n")
                for i, res_tuple in enumerate(top_n_for_score2_list):
                    p, s_1, t_1, s_2, t_2, c_s = res_tuple
                    f.write(f"{i+1}. Params: {p}\n")
                    f.write(f"   Score1: {s_1}, Alignment1: \"{t_1}\"\n")
                    f.write(f"   Score2: {s_2}, Alignment2: \"{t_2}\"\n")
                    f.write(f"   Combined Score: {c_s}\n\n")

                f.write(f"--- Top {N_TOP} Results for Combined Score (Score1 + Score2) (So Far) ---\n")
                for i, res_tuple in enumerate(top_n_for_combined_list):
                    p, s_1, t_1, s_2, t_2, c_s = res_tuple
                    f.write(f"{i+1}. Params: {p}\n")
                    f.write(f"   Score1: {s_1}, Alignment1: \"{t_1}\"\n")
                    f.write(f"   Score2: {s_2}, Alignment2: \"{t_2}\"\n")
                    f.write(f"   Combined Score: {c_s}\n\n")
        except Exception as e:
            print(f"Error writing to output file: {e}")


def main():
    global output_filename # Make it accessible for final save if needed
    # Pre-generate k-mer data
    # Or, ensure get_kmer_data is robust if called by children independently (it is with global cache)
    print("Initializing k-mer data for both datasets...")
    get_kmer_data(1, ref1, que1)
    get_kmer_data(2, ref2, que2)
    print("K-mer data initialization complete.")

    param_names = list(PARAM_RANGES.keys())
    param_value_lists = [PARAM_RANGES[name] for name in param_names]
    
    all_param_combinations = list(itertools.product(*param_value_lists))
    
    total_combs = len(all_param_combinations)
    print(f"Total parameter combinations to test: {total_combs}")
    
    num_processes = 16
    print(f"Using {num_processes} processes for parallel execution.")

    # Shared counter for progress, initialized for workers
    shared_progress_counter = multiprocessing.Value('i', 0)
    shared_total_combs = multiprocessing.Value('i', total_combs)


    start_time = time.time()
    
    processed_count = 0
    try:
        # Pass counter to worker initializer
        with multiprocessing.Pool(processes=num_processes, initializer=init_worker, initargs=(shared_progress_counter, shared_total_combs)) as pool:
            # Using imap_unordered to get results as they are completed
            for result_item in pool.imap_unordered(evaluate_params, all_param_combinations):
                if result_item:
                    params, s1, t1, s2, t2, cs = result_item
                    # Print to console (moved from worker for better synchronization of progress)
                    print(f"[Main - {shared_progress_counter.value}/{total_combs}] Params: {params}")
                    print(f"  Score1: {s1}, Alignment1: \"{t1}\"")
                    print(f"  Score2: {s2}, Alignment2: \"{t2}\"")
                    print(f"  Combined: {cs}")
                    
                    update_and_save_results(result_item, total_combs)
                    processed_count +=1
                # Optional: add a small sleep if I/O becomes a bottleneck, though unlikely
                # time.sleep(0.01) 

    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, attempting to save final results before terminating.")
    except Exception as e:
        print(f"An error occurred during processing: {e}")
    finally:
        print(f"Processed {processed_count} combinations.")
        # Ensure final save with whatever data has been processed
        # The update_and_save_results already saves, but an explicit call ensures the very last state is written.
        # However, if interrupted, the global lists might not have the absolute latest from workers not yet returned.
        # The current design saves after each result is processed by main, which is good.
        # A final call to save what's in the global lists is redundant if it's called after every result.
        # Let's ensure the file reflects the last processed item.
        # If the loop was interrupted, the last call to update_and_save_results would have saved the state.
        print("Ensuring final results are saved...")
        # No need to call update_and_save_results again here if it's called after every result.
        # The file should be up-to-date.

    end_time = time.time()
    print(f"Finished processing in {end_time - start_time:.2f} seconds.")
    print(f"Final results saved to {output_filename}")


if __name__ == "__main__":
    multiprocessing.freeze_support() 
    main()
