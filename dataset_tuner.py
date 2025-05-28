import numpy as np
from optimizer import AlignmentOptimizer, optimize_for_dataset
from run import function

class DatasetSpecificTuner:
    """Specialized tuner for different types of datasets"""
    
    def __init__(self):
        self.dataset_profiles = {
            'high_density': {
                'max_gap_param': [150, 200, 250],
                'max_diag_diff_param': [80, 120, 150],
                'overlap_factor_param': [0.2, 0.4, 0.6],
                'min_anchors_param': [2, 3, 4]
            },
            'low_density': {
                'max_gap_param': [300, 400, 500],
                'max_diag_diff_param': [200, 250, 300],
                'overlap_factor_param': [0.5, 0.7, 0.8],
                'min_anchors_param': [1, 2]
            },
            'mixed_strand': {
                'max_gap_param': [200, 250, 300],
                'max_diag_diff_param': [120, 150, 200],
                'overlap_factor_param': [0.3, 0.5, 0.7],
                'min_anchors_param': [1, 2, 3]
            },
            'single_strand': {
                'max_gap_param': [180, 220, 280],
                'max_diag_diff_param': [100, 140, 180],
                'overlap_factor_param': [0.4, 0.6, 0.8],
                'min_anchors_param': [1, 2]
            }
        }
    
    def classify_dataset(self, data):
        """Automatically classify dataset type"""
        n_anchors = len(data)
        
        # Calculate density
        if n_anchors < 2:
            return 'single_strand'
        
        query_positions = data[:, 0]
        query_span = np.max(query_positions) - np.min(query_positions)
        density = n_anchors / max(1, query_span)
        
        # Analyze strands
        strands = data[:, 2]
        unique_strands = len(np.unique(strands))
        
        # Classification logic
        if density > 0.05:
            dataset_type = 'high_density'
        else:
            dataset_type = 'low_density'
        
        if unique_strands > 1:
            if dataset_type == 'high_density':
                dataset_type = 'mixed_strand'
            # low_density already handles mixed strands well
        else:
            if dataset_type == 'low_density':
                dataset_type = 'single_strand'
        
        print(f"Dataset classified as: {dataset_type}")
        print(f"  Density: {density:.6f}, Strands: {unique_strands}, Anchors: {n_anchors}")
        
        return dataset_type
    
    def exhaustive_parameter_search(self, data, dataset_type=None):
        """Exhaustive search within dataset-specific parameter ranges"""
        if dataset_type is None:
            dataset_type = self.classify_dataset(data)
        
        if dataset_type not in self.dataset_profiles:
            print(f"Unknown dataset type: {dataset_type}, using mixed_strand profile")
            dataset_type = 'mixed_strand'
        
        param_ranges = self.dataset_profiles[dataset_type]
        
        optimizer = AlignmentOptimizer()
        base_params = {
            'max_gap_param': 250,
            'max_diag_diff_param': 150,
            'overlap_factor_param': 0.5,
            'min_anchors_param': 1
        }
        
        return optimizer.grid_search_fine_tuning(data, base_params, param_ranges)
    
    def progressive_refinement(self, data, max_rounds=3):
        """Progressive parameter refinement with multiple rounds"""
        print("Starting progressive refinement optimization...")
        
        # Initial classification and search
        dataset_type = self.classify_dataset(data)
        best_params, best_score, best_result = self.exhaustive_parameter_search(data, dataset_type)
        
        print(f"Round 1 - Best score: {best_score}")
        print(f"Round 1 - Best params: {best_params}")
        
        # Refine around best parameters
        for round_num in range(2, max_rounds + 1):
            print(f"\n--- Refinement Round {round_num} ---")
            
            # Create narrower search ranges around current best
            refined_ranges = self._create_refined_ranges(best_params)
            
            optimizer = AlignmentOptimizer()
            round_params, round_score, round_result = optimizer.grid_search_fine_tuning(
                data, best_params, refined_ranges
            )
            
            if round_score > best_score:
                best_params = round_params
                best_score = round_score
                best_result = round_result
                print(f"Round {round_num} - Improved! New score: {best_score}")
            else:
                print(f"Round {round_num} - No improvement, stopping")
                break
        
        return best_params, best_score, best_result
    
    def _create_refined_ranges(self, base_params):
        """Create refined parameter ranges around current best"""
        refined = {}
        
        for param, value in base_params.items():
            if param in ['max_gap_param', 'max_diag_diff_param']:
                step = max(10, int(value * 0.1))
                refined[param] = [
                    max(10, value - step),
                    value,
                    value + step
                ]
            elif param == 'overlap_factor_param':
                step = 0.1
                refined[param] = [
                    max(0.1, value - step),
                    value,
                    min(0.9, value + step)
                ]
            elif param == 'min_anchors_param':
                refined[param] = [
                    max(1, value - 1),
                    value,
                    value + 1
                ]
        
        return refined

def tune_two_datasets(dataset1, dataset2, names=None):
    """Specialized function to tune two different datasets"""
    if names is None:
        names = ['Dataset1', 'Dataset2']
    
    print("="*80)
    print("DUAL DATASET OPTIMIZATION")
    print("="*80)
    
    tuner = DatasetSpecificTuner()
    results = {}
    
    for i, (data, name) in enumerate(zip([dataset1, dataset2], names)):
        print(f"\n{'-'*60}")
        print(f"OPTIMIZING {name.upper()}")
        print(f"{'-'*60}")
        
        # Progressive refinement for each dataset
        params, score, result = tuner.progressive_refinement(data)
        
        results[name] = {
            'params': params,
            'score': score,
            'result': result,
            'dataset_type': tuner.classify_dataset(data)
        }
        
        print(f"\n{name} FINAL RESULTS:")
        print(f"  Score: {score}")
        print(f"  Parameters: {params}")
        print(f"  Result: {result}")
    
    # Compare results
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    for name, result_data in results.items():
        print(f"\n{name}:")
        print(f"  Dataset Type: {result_data['dataset_type']}")
        print(f"  Final Score: {result_data['score']}")
        print(f"  Optimal Parameters:")
        for param, value in result_data['params'].items():
            print(f"    {param}: {value}")
    
    return results

def quick_dataset_optimization(data, dataset_name="Dataset"):
    """Quick optimization for a single dataset"""
    print(f"Quick optimization for {dataset_name}")
    
    tuner = DatasetSpecificTuner()
    dataset_type = tuner.classify_dataset(data)
    
    # Use exhaustive search for quick results
    params, score, result = tuner.exhaustive_parameter_search(data, dataset_type)
    
    print(f"Quick optimization complete:")
    print(f"  Score: {score}")
    print(f"  Result: {result}")
    print(f"  Parameters: {params}")
    
    return params, result

if __name__ == "__main__":
    print("Dataset-Specific Tuner Ready!")
    print("Functions available:")
    print("- tune_two_datasets(data1, data2, ['name1', 'name2'])")
    print("- quick_dataset_optimization(data, 'dataset_name')")
    print("- DatasetSpecificTuner().progressive_refinement(data)")
