import json
import sys
from itertools import product
import os
import random
import math
from concurrent.futures import ThreadPoolExecutor
import threading
from multiprocessing import cpu_count

# Global thread-local storage for dataset references
thread_local_data = threading.local()

def initialize_thread_data(dataset_num):
    """Initialize thread-local data for each worker thread"""
    if not hasattr(thread_local_data, 'initialized'):
        from data import ref1, que1, ref2, que2
        from eval import calculate_value
        
        thread_local_data.ref1 = ref1
        thread_local_data.que1 = que1
        thread_local_data.ref2 = ref2
        thread_local_data.que2 = que2
        thread_local_data.calculate_value = calculate_value
        thread_local_data.initialized = True

def run_eval_threaded(dataset_num, alignment_result):
    """Thread-safe version of run_eval"""
    try:
        # Initialize thread-local data if needed
        initialize_thread_data(dataset_num)
        
        # Select the appropriate dataset
        if dataset_num == 1:
            ref, query = thread_local_data.ref1, thread_local_data.que1
        elif dataset_num == 2:
            ref, query = thread_local_data.ref2, thread_local_data.que2
        else:
            return 0
        
        # Convert alignment result to string format
        if isinstance(alignment_result, list):
            flat_values = []
            for tuple_item in alignment_result:
                if len(tuple_item) == 4:
                    flat_values.extend(tuple_item)
                else:
                    return 0
            tuples_str = ','.join(map(str, flat_values))
        else:
            tuples_str = str(alignment_result)
        
        # Calculate and return the score
        score = thread_local_data.calculate_value(tuples_str, ref, query)
        return score
        
    except Exception as e:
        print(f"ERROR in run_eval_threaded: {e}")
        return 0

def hill_climbing_search(dataset_num, initial_alignment, max_iterations=500, step_size=5):
    """Hill climbing optimization algorithm"""
    print(f"=== Hill Climbing Search ===")
    
    current_alignment = list(initial_alignment)
    current_score = run_eval(dataset_num, current_alignment)
    best_alignment = list(current_alignment)
    best_score = current_score
    
    print(f"Initial: {current_alignment} -> Score: {current_score}")
    
    for iteration in range(max_iterations):
        # Generate neighbors by modifying each segment
        neighbors = []
        
        for i in range(len(current_alignment)):
            q_start, q_end, r_start, r_end = current_alignment[i]
            
            # Generate neighbors by small modifications
            modifications = [
                (q_start + step_size, q_end, r_start, r_end),
                (q_start - step_size, q_end, r_start, r_end),
                (q_start, q_end + step_size, r_start, r_end),
                (q_start, q_end - step_size, r_start, r_end),
                (q_start, q_end, r_start + step_size, r_end),
                (q_start, q_end, r_start - step_size, r_end),
                (q_start, q_end, r_start, r_end + step_size),
                (q_start, q_end, r_start, r_end - step_size),
            ]
            
            for new_segment in modifications:
                new_q_start, new_q_end, new_r_start, new_r_end = new_segment
                # Validate segment
                if (new_q_start >= 0 and new_q_end > new_q_start and 
                    new_r_start >= 0 and new_r_end > new_r_start):
                    new_alignment = list(current_alignment)
                    new_alignment[i] = new_segment
                    neighbors.append(new_alignment)
        
        # Evaluate all neighbors and find the best one
        best_neighbor = None
        best_neighbor_score = current_score
        
        for j, neighbor in enumerate(neighbors):
            score = run_eval(dataset_num, neighbor)
            print(f"  Iter {iteration+1:3d}, Neighbor {j+1:2d}: {neighbor} -> Score: {score}")
            
            if score > best_neighbor_score:
                best_neighbor = neighbor
                best_neighbor_score = score
        
        # Move to best neighbor if it's better (hill climbing rule)
        if best_neighbor_score > current_score:
            current_alignment = best_neighbor
            current_score = best_neighbor_score
            print(f"  *** IMPROVEMENT: New current score: {current_score} ***")
            
            if current_score > best_score:
                best_alignment = list(current_alignment)
                best_score = current_score
                print(f"  *** NEW GLOBAL BEST: {best_score} ***")
        else:
            print(f"  No improvement found. Stopping hill climbing.")
            break
    
    return best_alignment, best_score

def simulated_annealing(dataset_num, initial_alignment, max_iterations=1000, initial_temp=1000, cooling_rate=0.95):
    """Simulated annealing optimization algorithm"""
    print(f"\n=== Simulated Annealing Search ===")
    
    current_alignment = list(initial_alignment)
    current_score = run_eval(dataset_num, current_alignment)
    best_alignment = list(current_alignment)
    best_score = current_score
    temperature = initial_temp
    
    print(f"Initial: {current_alignment} -> Score: {current_score}")
    print(f"Temperature: {temperature}, Cooling rate: {cooling_rate}")
    
    for iteration in range(max_iterations):
        # Generate random neighbor
        neighbor = list(current_alignment)
        segment_idx = random.randint(0, len(neighbor) - 1)
        q_start, q_end, r_start, r_end = neighbor[segment_idx]
        
        # Random modification with larger steps for exploration
        modifications = [
            random.randint(-15, 15),  # q_start change
            random.randint(-15, 15),  # q_end change  
            random.randint(-15, 15),  # r_start change
            random.randint(-15, 15),  # r_end change
        ]
        
        new_segment = (
            max(0, q_start + modifications[0]),
            max(q_start + 1, q_end + modifications[1]),
            max(0, r_start + modifications[2]),
            max(r_start + 1, r_end + modifications[3])
        )
        
        neighbor[segment_idx] = new_segment
        neighbor_score = run_eval(dataset_num, neighbor)
        
        # Accept or reject based on simulated annealing criteria
        delta = neighbor_score - current_score
        accept = False
        
        if delta > 0:
            accept = True
            reason = "BETTER"
        elif temperature > 0:
            probability = math.exp(delta / temperature)
            if random.random() < probability:
                accept = True
                reason = f"PROB({probability:.3f})"
            else:
                reason = f"REJECT({probability:.3f})"
        else:
            reason = "COLD_REJECT"
        
        print(f"  Iter {iteration+1:4d}: T={temperature:6.1f}, {neighbor} -> Score: {neighbor_score:5d} ({reason})")
        
        if accept:
            current_alignment = neighbor
            current_score = neighbor_score
            
            if current_score > best_score:
                best_alignment = list(current_alignment)
                best_score = current_score
                print(f"    *** NEW GLOBAL BEST: {best_score} ***")
        
        # Cool down
        temperature *= cooling_rate
        
        if iteration % 100 == 99:
            print(f"  Progress: {(iteration+1)/max_iterations*100:.1f}% | Current: {current_score} | Best: {best_score}")
    
    return best_alignment, best_score

def grid_search_around_point(dataset_num, initial_alignment, search_window=20, step=2):
    """Exhaustive grid search around initial point"""
    print(f"\n=== Grid Search (window={search_window}, step={step}) ===")
    
    best_score = run_eval(dataset_num, initial_alignment)
    best_alignment = initial_alignment
    
    print(f"Initial: {initial_alignment} -> Score: {best_score}")
    
    # For each alignment segment
    for i, (q_start, q_end, r_start, r_end) in enumerate(initial_alignment):
        print(f"\nOptimizing segment {i+1}: ({q_start}, {q_end}, {r_start}, {r_end})")
        
        # Define search ranges
        q_start_range = range(max(0, q_start - search_window), q_start + search_window + 1, step)
        q_end_range = range(max(q_start + 1, q_end - search_window), q_end + search_window + 1, step)
        r_start_range = range(max(0, r_start - search_window), r_start + search_window + 1, step)
        r_end_range = range(max(r_start + 1, r_end - search_window), r_end + search_window + 1, step)
        
        total_combinations = len(q_start_range) * len(q_end_range) * len(r_start_range) * len(r_end_range)
        tested = 0
        improvements_found = 0
        
        print(f"  Testing {total_combinations} combinations...")
        
        for new_q_start in q_start_range:
            for new_q_end in q_end_range:
                if new_q_end <= new_q_start:
                    continue
                for new_r_start in r_start_range:
                    for new_r_end in r_end_range:
                        if new_r_end <= new_r_start:
                            continue
                        
                        tested += 1
                        
                        # Create new alignment with modified segment
                        new_alignment = list(best_alignment)
                        new_alignment[i] = (new_q_start, new_q_end, new_r_start, new_r_end)
                        
                        score = run_eval(dataset_num, new_alignment)
                        
                        # Display every search attempt
                        print(f"  Search {tested:4d}/{total_combinations}: {new_alignment} -> Score: {score}")
                        
                        if score > best_score:
                            best_score = score
                            best_alignment = new_alignment
                            improvements_found += 1
                            print(f"    *** NEW BEST SCORE: {score} ***")
        
        if improvements_found == 0:
            print(f"  No improvements found for segment {i+1}.")
        else:
            print(f"  Improvements found for segment {i+1}: {improvements_found}")
    
    print(f"\nBest alignment after grid search: {best_alignment} -> Score: {best_score}")
    
    return best_alignment, best_score

def parallel_grid_search(dataset_num, initial_alignment, search_window=20, step=2, num_threads=16):
    """Parallel grid search using multiple threads"""
    print(f"\n=== Parallel Grid Search (window={search_window}, step={step}, threads={num_threads}) ===")
    
    best_score = run_eval_threaded(dataset_num, initial_alignment)
    best_alignment = initial_alignment
    
    print(f"Initial: {initial_alignment} -> Score: {best_score}")
    
    # For each alignment segment
    for i, (q_start, q_end, r_start, r_end) in enumerate(initial_alignment):
        print(f"\nOptimizing segment {i+1}: ({q_start}, {q_end}, {r_start}, {r_end})")
        
        # Define search ranges
        q_start_range = range(max(0, q_start - search_window), q_start + search_window + 1, step)
        q_end_range = range(max(q_start + 1, q_end - search_window), q_end + search_window + 1, step)
        r_start_range = range(max(0, r_start - search_window), r_start + search_window + 1, step)
        r_end_range = range(max(r_start + 1, r_end - search_window), r_end + search_window + 1, step)
        
        # Generate all combinations
        combinations = []
        for new_q_start in q_start_range:
            for new_q_end in q_end_range:
                if new_q_end <= new_q_start:
                    continue
                for new_r_start in r_start_range:
                    for new_r_end in r_end_range:
                        if new_r_end <= new_r_start:
                            continue
                        
                        new_alignment = list(best_alignment)
                        new_alignment[i] = (new_q_start, new_q_end, new_r_start, new_r_end)
                        combinations.append(new_alignment)
        
        print(f"  Testing {len(combinations)} combinations using {num_threads} threads...")
        
        # Function to evaluate a batch of alignments
        def evaluate_alignment(alignment):
            score = run_eval_threaded(dataset_num, alignment)
            return alignment, score
        
        improvements_found = 0
        tested = 0
        
        # Use ThreadPoolExecutor to parallelize evaluations
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit all jobs
            futures = [executor.submit(evaluate_alignment, combo) for combo in combinations]
            
            # Process results as they complete
            for future in futures:
                tested += 1
                alignment, score = future.result()
                
                print(f"  Search {tested:4d}/{len(combinations)}: {alignment} -> Score: {score}")
                
                if score > best_score:
                    best_score = score
                    best_alignment = alignment
                    improvements_found += 1
                    print(f"    *** NEW BEST SCORE: {score} ***")
        
        if improvements_found == 0:
            print(f"  No improvements found for segment {i+1}.")
        else:
            print(f"  Improvements found for segment {i+1}: {improvements_found}")
    
    print(f"\nBest alignment after parallel grid search: {best_alignment} -> Score: {best_score}")
    return best_alignment, best_score

def parallel_hill_climbing(dataset_num, initial_alignment, max_iterations=500, step_size=5, num_threads=16):
    """Parallel hill climbing using multiple threads"""
    print(f"=== Parallel Hill Climbing Search (threads={num_threads}) ===")
    
    current_alignment = list(initial_alignment)
    current_score = run_eval_threaded(dataset_num, current_alignment)
    best_alignment = list(current_alignment)
    best_score = current_score
    
    print(f"Initial: {current_alignment} -> Score: {current_score}")
    
    for iteration in range(max_iterations):
        # Generate all neighbors
        neighbors = []
        
        for i in range(len(current_alignment)):
            q_start, q_end, r_start, r_end = current_alignment[i]
            
            modifications = [
                (q_start + step_size, q_end, r_start, r_end),
                (q_start - step_size, q_end, r_start, r_end),
                (q_start, q_end + step_size, r_start, r_end),
                (q_start, q_end - step_size, r_start, r_end),
                (q_start, q_end, r_start + step_size, r_end),
                (q_start, q_end, r_start - step_size, r_end),
                (q_start, q_end, r_start, r_end + step_size),
                (q_start, q_end, r_start, r_end - step_size),
            ]
            
            for new_segment in modifications:
                new_q_start, new_q_end, new_r_start, new_r_end = new_segment
                if (new_q_start >= 0 and new_q_end > new_q_start and 
                    new_r_start >= 0 and new_r_end > new_r_start):
                    new_alignment = list(current_alignment)
                    new_alignment[i] = new_segment
                    neighbors.append(new_alignment)
        
        # Evaluate all neighbors in parallel
        def evaluate_neighbor(neighbor):
            score = run_eval_threaded(dataset_num, neighbor)
            return neighbor, score
        
        best_neighbor = None
        best_neighbor_score = current_score
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(evaluate_neighbor, neighbor) for neighbor in neighbors]
            
            for j, future in enumerate(futures):
                neighbor, score = future.result()
                print(f"  Iter {iteration+1:3d}, Neighbor {j+1:2d}: {neighbor} -> Score: {score}")
                
                if score > best_neighbor_score:
                    best_neighbor = neighbor
                    best_neighbor_score = score
        
        # Move to best neighbor if it's better
        if best_neighbor_score > current_score:
            current_alignment = best_neighbor
            current_score = best_neighbor_score
            print(f"  *** IMPROVEMENT: New current score: {current_score} ***")
            
            if current_score > best_score:
                best_alignment = list(current_alignment)
                best_score = current_score
                print(f"  *** NEW GLOBAL BEST: {best_score} ***")
        else:
            print(f"  No improvement found. Stopping hill climbing.")
            break
    
    return best_alignment, best_score

def parallel_beam_search(dataset_num, initial_alignment, beam_width=5, max_iterations=50, num_threads=16):
    """Parallel beam search using multiple threads"""
    print(f"\n=== Parallel Beam Search (beam_width={beam_width}, threads={num_threads}) ===")
    
    # Initialize beam with the starting alignment
    beam = [(initial_alignment, run_eval_threaded(dataset_num, initial_alignment))]
    print(f"Initial: {initial_alignment} -> Score: {beam[0][1]}")
    
    best_alignment, best_score = beam[0]
    
    for iteration in range(max_iterations):
        # Generate all candidates from current beam
        all_candidates = []
        
        for alignment, score in beam:
            for i in range(len(alignment)):
                q_start, q_end, r_start, r_end = alignment[i]
                
                for step in [1, 2, 3, 5]:
                    modifications = [
                        (q_start + step, q_end, r_start, r_end),
                        (q_start - step, q_end, r_start, r_end),
                        (q_start, q_end + step, r_start, r_end),
                        (q_start, q_end - step, r_start, r_end),
                        (q_start, q_end, r_start + step, r_end),
                        (q_start, q_end, r_start - step, r_end),
                        (q_start, q_end, r_start, r_end + step),
                        (q_start, q_end, r_start, r_end - step),
                    ]
                    
                    for new_segment in modifications:
                        new_q_start, new_q_end, new_r_start, new_r_end = new_segment
                        if (new_q_start >= 0 and new_q_end > new_q_start and 
                            new_r_start >= 0 and new_r_end > new_r_start):
                            new_alignment = list(alignment)
                            new_alignment[i] = new_segment
                            all_candidates.append(new_alignment)
        
        # Evaluate all candidates in parallel
        def evaluate_candidate(candidate):
            score = run_eval_threaded(dataset_num, candidate)
            return candidate, score
        
        evaluated_candidates = []
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(evaluate_candidate, candidate) for candidate in all_candidates]
            
            for future in futures:
                alignment, score = future.result()
                print(f"  Iter {iteration+1:2d}: {alignment} -> Score: {score}")
                evaluated_candidates.append((alignment, score))
                
                if score > best_score:
                    best_alignment = alignment
                    best_score = score
                    print(f"    *** NEW BEST SCORE: {best_score} ***")
        
        # Keep only the top beam_width candidates
        all_candidates_with_scores = beam + evaluated_candidates
        all_candidates_with_scores.sort(key=lambda x: x[1], reverse=True)
        beam = all_candidates_with_scores[:beam_width]
        
        print(f"  Top {len(beam)} candidates after iteration {iteration+1}:")
        for j, (align, sc) in enumerate(beam):
            print(f"    {j+1}: Score {sc}")
        
        if beam[0][1] <= best_score and iteration > 5:
            print(f"  No improvement in beam. Stopping.")
            break
    
    return best_alignment, best_score

def optimize_multiple_datasets():
    """Optimize alignments for multiple datasets using parallel algorithms"""
    
    # Initial results from the prompt
    datasets = {
        1: {
            'initial_alignment': [(0, 2039, 0, 2039), (2039, 21493, 8335, 27802), (21496, 29827, 21548, 29828)],
            'initial_score': 27074
        },
        2: {
            'initial_alignment': [(0, 2020, 0, 1564), (2059, 2068, 696, 855), (2299, 2500, 1054, 1700)],
            'initial_score': 0
        }
    }
    
    num_threads = 16
    print(f"Using {num_threads} threads for parallel optimization")
    
    results = {}
    
    for dataset_num, data in datasets.items():
        print(f"\n{'='*60}")
        print(f"OPTIMIZING DATASET {dataset_num}")
        print(f"{'='*60}")
        print(f"Initial alignment: {data['initial_alignment']}")
        print(f"Initial score: {data['initial_score']}")
        
        # Try different parallel optimization algorithms
        algorithms = []
        
        # 1. Parallel Beam Search
        print(f"\n{'-'*40}")
        print("ALGORITHM 1: PARALLEL BEAM SEARCH")
        print(f"{'-'*40}")
        beam_alignment, beam_score = parallel_beam_search(
            dataset_num, data['initial_alignment'], beam_width=3, max_iterations=15, num_threads=num_threads
        )
        algorithms.append(('Parallel Beam Search', beam_alignment, beam_score))
        
        # 2. Parallel Hill Climbing
        print(f"\n{'-'*40}")
        print("ALGORITHM 2: PARALLEL HILL CLIMBING")
        print(f"{'-'*40}")
        hill_alignment, hill_score = parallel_hill_climbing(
            dataset_num, data['initial_alignment'], max_iterations=20, step_size=2, num_threads=num_threads
        )
        algorithms.append(('Parallel Hill Climbing', hill_alignment, hill_score))
        
        # 3. Parallel Grid Search
        print(f"\n{'-'*40}")
        print("ALGORITHM 3: PARALLEL GRID SEARCH")
        print(f"{'-'*40}")
        grid_alignment, grid_score = parallel_grid_search(
            dataset_num, data['initial_alignment'], search_window=8, step=2, num_threads=num_threads
        )
        algorithms.append(('Parallel Grid Search', grid_alignment, grid_score))
        
        # Find best result
        best_algorithm, best_alignment, best_score = max(algorithms, key=lambda x: x[2])
        
        results[dataset_num] = {
            'initial_alignment': data['initial_alignment'],
            'initial_score': data['initial_score'],
            'algorithms': algorithms,
            'best_algorithm': best_algorithm,
            'best_alignment': best_alignment,
            'best_score': best_score,
            'improvement': best_score - data['initial_score']
        }
        
        print(f"\n{'='*50}")
        print(f"DATASET {dataset_num} SUMMARY")
        print(f"{'='*50}")
        print(f"Initial:        {data['initial_alignment']} -> Score: {data['initial_score']}")
        for alg_name, alg_alignment, alg_score in algorithms:
            improvement = alg_score - data['initial_score']
            print(f"{alg_name:25s}: Score {alg_score} (+{improvement})")
        print(f"BEST RESULT:    {best_algorithm} with score {best_score}")
    
    return results

if __name__ == "__main__":
    # Check if required files exist
    required_files = ['eval.py', 'data.py']
    for file in required_files:
        if not os.path.exists(file):
            print(f"Error: {file} not found in current directory")
            sys.exit(1)
    
    print("Starting parallel alignment optimization with 16 threads...")
    print(f"Working directory: {os.getcwd()}")
    print(f"Available CPU cores: {cpu_count()}")
    
    results = optimize_multiple_datasets()
    
    # Print final summary
    print(f"\n{'='*80}")
    print("FINAL PARALLEL OPTIMIZATION SUMMARY")
    print(f"{'='*80}")
    
    for dataset_num, result in results.items():
        print(f"\nDataset {dataset_num}:")
        print(f"  Initial Score:     {result['initial_score']}")
        print(f"  Best Algorithm:    {result['best_algorithm']}")
        print(f"  Best Score:        {result['best_score']}")
        print(f"  Total Improvement: +{result['improvement']}")
        print(f"  Best Alignment:    {result['best_alignment']}")
        
        print("  Algorithm Comparison:")
        for alg_name, alg_alignment, alg_score in result['algorithms']:
            improvement = alg_score - result['initial_score']
            print(f"    {alg_name:25s}: {alg_score:6d} (+{improvement:4d})")
    
    print(f"\nParallel optimization complete!")