import numpy as np
from run import function
from eval import seq2hashtable_multi_test, calculate_value
from data import ref1, que1, ref2, que2

# 1. Define Sample Data
# TODO: Replace this with your representative sample data for effective tuning.
# Format: [[q_s, r_s, strand, kmersize], ...]
sample_data = np.array([
    [10, 100, 1, 20],
    [40, 135, 1, 20],    # Chainable with first
    [50, 200, -1, 20],   # Different strand
    [90, 185, 1, 20],    # Potentially chainable with second
    [120, 220, 1, 20],   # Chainable with fourth
    [200, 500, 1, 20],   # Separate chain
    [230, 532, 1, 20]    # Chainable with previous
])

# 2. Define Initial Parameters and Step Sizes
initial_params = {
    'max_gap_param': 250,
    'max_diag_diff_param': 150,
    'overlap_factor_param': 0.5,
    'min_anchors_param': 2
}

step_sizes = {
    'max_gap_param': 50,          # Step for adjusting max_gap_param
    'max_diag_diff_param': 25,    # Step for adjusting max_diag_diff_param
    'overlap_factor_param': 0.1,  # Step for adjusting overlap_factor_param
    'min_anchors_param': 1        # Step for adjusting min_anchors_param
}

# Parameter constraints
param_constraints = {
    'max_gap_param': {'min': 10, 'max': 1000, 'type': int},
    'max_diag_diff_param': {'min': 10, 'max': 500, 'type': int},
    'overlap_factor_param': {'min': 0.1, 'max': 0.9, 'type': float},
    'min_anchors_param': {'min': 1, 'max': 10, 'type': int}
}

# 3. Evaluation Function
def evaluate_output(params):
    """
    Evaluates the performance of function with given parameters on both datasets.
    Returns the combined score from both datasets.
    """
    total_score = 0
    
    # Process Dataset 1
    try:
        data1 = seq2hashtable_multi_test(ref1, que1, kmersize=9, shift=1)
        output_str1 = function(data1, **params)
        score1 = calculate_value(output_str1, ref1, que1)
        total_score += score1
    except Exception as e:
        print(f"Error processing dataset 1: {e}")
    
    # Process Dataset 2
    try:
        data2 = seq2hashtable_multi_test(ref2, que2, kmersize=9, shift=1)
        output_str2 = function(data2, **params)
        score2 = calculate_value(output_str2, ref2, que2)
        total_score += score2
    except Exception as e:
        print(f"Error processing dataset 2: {e}")
    
    return total_score

# 4. Parameter Tuning Loop (Hill Climbing)
def tune_parameters():
    current_params = initial_params.copy()
    
    # Run initial evaluation
    try:
        current_score = evaluate_output(current_params)
        print(f"Initial parameters: {current_params}")
        print(f"Initial score: {current_score}")
    except Exception as e:
        print(f"Error with initial params: {e}")
        return
    
    best_params = current_params.copy()
    best_score = current_score
    
    max_iterations = 30  # Maximum number of iterations
    no_improvement_count = 0
    max_no_improvement = 5  # Stop after this many iterations without improvement
    
    for iteration in range(max_iterations):
        improved = False
        
        # Try adjusting each parameter
        for param_name in current_params:
            # Try increasing the parameter
            test_params = current_params.copy()
            test_params[param_name] += step_sizes[param_name]
            
            # Apply constraints
            constraint = param_constraints[param_name]
            test_params[param_name] = min(max(test_params[param_name], constraint['min']), constraint['max'])
            test_params[param_name] = constraint['type'](test_params[param_name])
            
            try:
                score = evaluate_output(test_params)
                
                if score > best_score:
                    best_score = score
                    best_params = test_params.copy()
                    current_params = test_params.copy()
                    improved = True
                    print(f"Iteration {iteration+1}, improved by increasing {param_name} to {test_params[param_name]}")
                    print(f"New score: {best_score}, New params: {best_params}")
                    # Continue to next parameter from this new best position
                    continue
            except Exception as e:
                print(f"Error when increasing {param_name} to {test_params[param_name]}: {e}")
            
            # Try decreasing the parameter
            test_params = current_params.copy()
            test_params[param_name] -= step_sizes[param_name]
            
            # Apply constraints
            constraint = param_constraints[param_name]
            test_params[param_name] = min(max(test_params[param_name], constraint['min']), constraint['max'])
            test_params[param_name] = constraint['type'](test_params[param_name])
            
            try:
                score = evaluate_output(test_params)
                
                if score > best_score:
                    best_score = score
                    best_params = test_params.copy()
                    current_params = test_params.copy()
                    improved = True
                    print(f"Iteration {iteration+1}, improved by decreasing {param_name} to {test_params[param_name]}")
                    print(f"New score: {best_score}, New params: {best_params}")
                    # Continue to next parameter from this new best position
                    continue
            except Exception as e:
                print(f"Error when decreasing {param_name} to {test_params[param_name]}: {e}")
        
        # Check if we improved in this iteration
        if not improved:
            no_improvement_count += 1
            print(f"Iteration {iteration+1}, no improvement. ({no_improvement_count}/{max_no_improvement})")
            
            # Optionally, reduce step sizes if no improvement
            if no_improvement_count % 2 == 0:
                for param in step_sizes:
                    step_sizes[param] = max(step_sizes[param] * 0.5, 1 if param_constraints[param]['type'] == int else 0.05)
                print(f"Reduced step sizes: {step_sizes}")
            
            if no_improvement_count >= max_no_improvement:
                print(f"Stopping: No improvement for {max_no_improvement} iterations")
                break
        else:
            no_improvement_count = 0

    # 5. Print Best Results and Final Evaluation
    print("\nParameter tuning finished.")
    print(f"Best score: {best_score}")
    print(f"Best parameters found: {best_params}")
    
    # Evaluate final results on both datasets
    try:
        data1 = seq2hashtable_multi_test(ref1, que1, kmersize=9, shift=1)
        output_str1 = function(data1, **best_params)
        score1 = calculate_value(output_str1, ref1, que1)
        print(f"Dataset 1 final score: {score1}")
        
        data2 = seq2hashtable_multi_test(ref2, que2, kmersize=9, shift=1)
        output_str2 = function(data2, **best_params)
        score2 = calculate_value(output_str2, ref2, que2)
        print(f"Dataset 2 final score: {score2}")
        
        print(f"Total final score: {score1 + score2}")
    except Exception as e:
        print(f"Error in final evaluation: {e}")

if __name__ == '__main__':
    tune_parameters()
