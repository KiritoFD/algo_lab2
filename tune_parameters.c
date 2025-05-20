#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

// Forward declarations from run.c
void chain_anchors_kernel(int n_anchors, 
                          int* q_s_arr, int* q_e_arr, 
                          int* r_s_arr, int* r_e_arr, 
                          char* strand_arr, 
                          int kmersize, 
                          int max_gap_between_anchors, 
                          int max_diagonal_difference, 
                          int max_allowed_overlap,
                          int* dp_score,
                          int* parent_idx);

int select_segments_kernel(int n_segs, 
                           int* seg_q_s_arr, 
                           int* seg_q_e_arr, 
                           int* seg_scores_arr,
                           int* dp_select_score,
                           int* prev_select_idx,
                           int* selected_indices);

// Forward declarations from eval.c
int calculate_value(const char* tuples_str, const char* ref, const char* query);
void format_tuples_for_display(const char* tuples_str);

// Sample data for tuning (similar to Python version)
int sample_data[][4] = {
    {10, 100, 1, 20},
    {40, 135, 1, 20},    // Chainable with first
    {50, 200, -1, 20},   // Different strand
    {90, 185, 1, 20},    // Potentially chainable with second
    {120, 220, 1, 20},   // Chainable with fourth
    {200, 500, 1, 20},   // Separate chain
    {230, 532, 1, 20}    // Chainable with previous
};
const int sample_data_size = 7; // Number of rows in sample_data

// Parameter structure
typedef struct {
    int max_gap_param;
    int max_diag_diff_param;
    float overlap_factor_param;
    int min_anchors_param;
} Parameters;

// Parameter constraints
typedef struct {
    int min;
    int max;
} IntConstraint;

typedef struct {
    float min;
    float max;
} FloatConstraint;

// Define parameter constraints
IntConstraint max_gap_constraint = {10, 1000};
IntConstraint max_diag_diff_constraint = {10, 500};
FloatConstraint overlap_factor_constraint = {0.1f, 0.9f};
IntConstraint min_anchors_constraint = {1, 10};

// Step sizes for parameter tuning
typedef struct {
    int max_gap_step;
    int max_diag_diff_step;
    float overlap_factor_step;
    int min_anchors_step;
} StepSizes;

// Implementation of a simplified version of the Python function
char* function(int** data, int data_rows, Parameters params) {
    if (data_rows == 0) {
        return strdup("");
    }

    int kmersize = data[0][3];
    if (kmersize <= 0) {
        return strdup("");
    }

    // Create arrays for the chain_anchors_kernel
    int* q_s_arr = (int*)malloc(data_rows * sizeof(int));
    int* q_e_arr = (int*)malloc(data_rows * sizeof(int));
    int* r_s_arr = (int*)malloc(data_rows * sizeof(int));
    int* r_e_arr = (int*)malloc(data_rows * sizeof(int));
    char* strand_arr = (char*)malloc(data_rows * sizeof(char));

    // Python-like anchors (for later reconstruction)
    typedef struct {
        int q_s, q_e, r_s, r_e;
        char strand;
        int id;
    } Anchor;

    Anchor* py_anchors = (Anchor*)malloc(data_rows * sizeof(Anchor));

    // Fill arrays
    for (int i = 0; i < data_rows; i++) {
        int q_s = data[i][0];
        int r_s = data[i][1];
        char strand_val = (char)data[i][2];
        
        q_s_arr[i] = q_s;
        q_e_arr[i] = q_s + kmersize;
        r_s_arr[i] = r_s;
        r_e_arr[i] = r_s + kmersize;
        strand_arr[i] = strand_val;
        
        py_anchors[i].q_s = q_s;
        py_anchors[i].q_e = q_s + kmersize;
        py_anchors[i].r_s = r_s;
        py_anchors[i].r_e = r_s + kmersize;
        py_anchors[i].strand = strand_val;
        py_anchors[i].id = i;
    }

    // Sort py_anchors by q_s, r_s (simplified sorting)
    // Bubble sort for simplicity - not efficient for large datasets
    for (int i = 0; i < data_rows - 1; i++) {
        for (int j = 0; j < data_rows - i - 1; j++) {
            if (py_anchors[j].q_s > py_anchors[j + 1].q_s || 
                (py_anchors[j].q_s == py_anchors[j + 1].q_s && 
                 py_anchors[j].r_s > py_anchors[j + 1].r_s)) {
                Anchor temp = py_anchors[j];
                py_anchors[j] = py_anchors[j + 1];
                py_anchors[j + 1] = temp;
            }
        }
    }

    // Update arrays based on sorted anchors
    for (int i = 0; i < data_rows; i++) {
        q_s_arr[i] = py_anchors[i].q_s;
        q_e_arr[i] = py_anchors[i].q_e;
        r_s_arr[i] = py_anchors[i].r_s;
        r_e_arr[i] = py_anchors[i].r_e;
        strand_arr[i] = py_anchors[i].strand;
    }

    // Call chain_anchors_kernel
    int* dp_score = (int*)malloc(data_rows * sizeof(int));
    int* parent_idx = (int*)malloc(data_rows * sizeof(int));
    
    int max_allowed_overlap = (int)(kmersize * params.overlap_factor_param);
    
    chain_anchors_kernel(data_rows, q_s_arr, q_e_arr, r_s_arr, r_e_arr, strand_arr,
                         kmersize, params.max_gap_param, params.max_diag_diff_param, 
                         max_allowed_overlap, dp_score, parent_idx);

    // Form candidate segments
    typedef struct {
        int q_s, q_e, r_s, r_e;
        int score;
        char strand;
    } Segment;

    Segment* candidate_segments = NULL;
    int candidate_segments_count = 0;

    for (int i = 0; i < data_rows; i++) {
        // Trace back to form chain
        int* chain_indices = (int*)malloc(data_rows * sizeof(int));
        int chain_length = 0;
        int curr = i;
        
        while (curr != -1) {
            chain_indices[chain_length++] = curr;
            curr = parent_idx[curr];
        }

        // Check if chain meets minimum length requirement
        if (chain_length >= params.min_anchors_param) {
            // Reverse the chain
            for (int j = 0; j < chain_length / 2; j++) {
                int temp = chain_indices[j];
                chain_indices[j] = chain_indices[chain_length - j - 1];
                chain_indices[chain_length - j - 1] = temp;
            }

            int first_idx = chain_indices[0];
            int last_idx = chain_indices[chain_length - 1];

            int q_start = py_anchors[first_idx].q_s;
            int q_end = py_anchors[last_idx].q_e;

            // Find min r_start and max r_end
            int r_start = py_anchors[chain_indices[0]].r_s;
            int r_end = py_anchors[chain_indices[0]].r_e;

            for (int j = 1; j < chain_length; j++) {
                int idx = chain_indices[j];
                if (py_anchors[idx].r_s < r_start) r_start = py_anchors[idx].r_s;
                if (py_anchors[idx].r_e > r_end) r_end = py_anchors[idx].r_e;
            }

            // Add segment to candidate list
            candidate_segments_count++;
            candidate_segments = (Segment*)realloc(candidate_segments, 
                                                  candidate_segments_count * sizeof(Segment));
            
            candidate_segments[candidate_segments_count - 1].q_s = q_start;
            candidate_segments[candidate_segments_count - 1].q_e = q_end;
            candidate_segments[candidate_segments_count - 1].r_s = r_start;
            candidate_segments[candidate_segments_count - 1].r_e = r_end;
            candidate_segments[candidate_segments_count - 1].score = dp_score[i];
            candidate_segments[candidate_segments_count - 1].strand = py_anchors[i].strand;
        }

        free(chain_indices);
    }

    // Sort segments by q_s
    for (int i = 0; i < candidate_segments_count - 1; i++) {
        for (int j = 0; j < candidate_segments_count - i - 1; j++) {
            if (candidate_segments[j].q_s > candidate_segments[j + 1].q_s) {
                Segment temp = candidate_segments[j];
                candidate_segments[j] = candidate_segments[j + 1];
                candidate_segments[j + 1] = temp;
            }
        }
    }

    // Select non-overlapping segments
    int* seg_q_s_arr = (int*)malloc(candidate_segments_count * sizeof(int));
    int* seg_q_e_arr = (int*)malloc(candidate_segments_count * sizeof(int));
    int* seg_scores_arr = (int*)malloc(candidate_segments_count * sizeof(int));
    
    for (int i = 0; i < candidate_segments_count; i++) {
        seg_q_s_arr[i] = candidate_segments[i].q_s;
        seg_q_e_arr[i] = candidate_segments[i].q_e;
        seg_scores_arr[i] = candidate_segments[i].score;
    }

    int* dp_select_score = (int*)malloc(candidate_segments_count * sizeof(int));
    int* prev_select_idx = (int*)malloc(candidate_segments_count * sizeof(int));
    int* selected_indices = (int*)malloc(candidate_segments_count * sizeof(int));

    int selected_count = select_segments_kernel(
        candidate_segments_count, seg_q_s_arr, seg_q_e_arr, seg_scores_arr,
        dp_select_score, prev_select_idx, selected_indices);

    // Format output
    char* output = (char*)malloc(1);  // Start with empty string
    output[0] = '\0';
    
    for (int i = selected_count - 1; i >= 0; i--) {  // Reverse order
        int idx = selected_indices[i];
        char buffer[100];
        
        // First segment doesn't need a comma prefix
        if (i == selected_count - 1) {
            sprintf(buffer, "%d,%d,%d,%d", 
                    candidate_segments[idx].q_s,
                    candidate_segments[idx].q_e,
                    candidate_segments[idx].r_s,
                    candidate_segments[idx].r_e);
        } else {
            sprintf(buffer, ",%d,%d,%d,%d", 
                    candidate_segments[idx].q_s,
                    candidate_segments[idx].q_e,
                    candidate_segments[idx].r_s,
                    candidate_segments[idx].r_e);
        }
        
        size_t current_len = strlen(output);
        size_t buffer_len = strlen(buffer);
        output = (char*)realloc(output, current_len + buffer_len + 1);
        strcat(output, buffer);
    }

    // Clean up
    free(q_s_arr);
    free(q_e_arr);
    free(r_s_arr);
    free(r_e_arr);
    free(strand_arr);
    free(py_anchors);
    free(dp_score);
    free(parent_idx);
    free(candidate_segments);
    free(seg_q_s_arr);
    free(seg_q_e_arr);
    free(seg_scores_arr);
    free(dp_select_score);
    free(prev_select_idx);
    free(selected_indices);

    return output;
}

// Evaluate function output with given parameters
float evaluate_params(Parameters params, const char* ref1, const char* query1, 
                     const char* ref2, const char* query2) {
    float total_score = 0.0f;
    
    // This is a simplified placeholder for seq2hashtable_multi_test
    // In a full implementation, you would need to implement this function
    // or link to it from a separate module
    
    // For demonstration, we'll use the sample data
    int** sample_data_copy = (int**)malloc(sample_data_size * sizeof(int*));
    for (int i = 0; i < sample_data_size; i++) {
        sample_data_copy[i] = (int*)malloc(4 * sizeof(int));
        memcpy(sample_data_copy[i], sample_data[i], 4 * sizeof(int));
    }
    
    char* output = function(sample_data_copy, sample_data_size, params);
    
    // Clean up sample data copy
    for (int i = 0; i < sample_data_size; i++) {
        free(sample_data_copy[i]);
    }
    free(sample_data_copy);
    
    // In a real implementation, we would call:
    // float score1 = calculate_value(output, ref1, query1);
    // float score2 = calculate_value(output, ref2, query2);
    // total_score = score1 + score2;
    
    // For demonstration, just return a simulated score
    // This would be replaced with actual scoring in a full implementation
    
    // Clean up
    free(output);
    
    // Return a simulated score
    return params.max_gap_param * 0.1f + params.max_diag_diff_param * 0.05f - 
           params.overlap_factor_param * 10.0f + params.min_anchors_param * 5.0f;
}

// Apply constraints to parameters
void apply_constraints(Parameters* params) {
    // Apply integer constraints
    if (params->max_gap_param < max_gap_constraint.min) 
        params->max_gap_param = max_gap_constraint.min;
    if (params->max_gap_param > max_gap_constraint.max) 
        params->max_gap_param = max_gap_constraint.max;
    
    if (params->max_diag_diff_param < max_diag_diff_constraint.min) 
        params->max_diag_diff_param = max_diag_diff_constraint.min;
    if (params->max_diag_diff_param > max_diag_diff_constraint.max) 
        params->max_diag_diff_param = max_diag_diff_constraint.max;
    
    // Apply float constraint
    if (params->overlap_factor_param < overlap_factor_constraint.min) 
        params->overlap_factor_param = overlap_factor_constraint.min;
    if (params->overlap_factor_param > overlap_factor_constraint.max) 
        params->overlap_factor_param = overlap_factor_constraint.max;
    
    if (params->min_anchors_param < min_anchors_constraint.min) 
        params->min_anchors_param = min_anchors_constraint.min;
    if (params->min_anchors_param > min_anchors_constraint.max) 
        params->min_anchors_param = min_anchors_constraint.max;
}

// Hill climbing algorithm for parameter tuning
void tune_parameters(const char* ref1, const char* query1, const char* ref2, const char* query2) {
    // Initial parameters
    Parameters current_params = {
        .max_gap_param = 250,
        .max_diag_diff_param = 150,
        .overlap_factor_param = 0.5f,
        .min_anchors_param = 2
    };
    
    // Step sizes
    StepSizes step_sizes = {
        .max_gap_step = 50,
        .max_diag_diff_step = 25,
        .overlap_factor_step = 0.1f,
        .min_anchors_step = 1
    };
    
    // Evaluate initial parameters
    float current_score = evaluate_params(current_params, ref1, query1, ref2, query2);
    printf("Initial parameters: max_gap=%d, max_diag_diff=%d, overlap_factor=%.1f, min_anchors=%d\n",
           current_params.max_gap_param, current_params.max_diag_diff_param,
           current_params.overlap_factor_param, current_params.min_anchors_param);
    printf("Initial score: %.2f\n", current_score);
    
    Parameters best_params = current_params;
    float best_score = current_score;
    
    const int max_iterations = 30;
    int no_improvement_count = 0;
    const int max_no_improvement = 5;
    
    for (int iteration = 0; iteration < max_iterations; iteration++) {
        bool improved = false;
        
        // Try adjusting each parameter
        
        // 1. Try increasing/decreasing max_gap_param
        Parameters test_params = current_params;
        test_params.max_gap_param += step_sizes.max_gap_step;
        apply_constraints(&test_params);
        
        float score = evaluate_params(test_params, ref1, query1, ref2, query2);
        if (score > best_score) {
            best_score = score;
            best_params = test_params;
            current_params = test_params;
            improved = true;
            printf("Iteration %d, improved by increasing max_gap_param to %d\n", 
                   iteration + 1, test_params.max_gap_param);
            printf("New score: %.2f\n", best_score);
            continue;
        }
        
        test_params = current_params;
        test_params.max_gap_param -= step_sizes.max_gap_step;
        apply_constraints(&test_params);
        
        score = evaluate_params(test_params, ref1, query1, ref2, query2);
        if (score > best_score) {
            best_score = score;
            best_params = test_params;
            current_params = test_params;
            improved = true;
            printf("Iteration %d, improved by decreasing max_gap_param to %d\n", 
                   iteration + 1, test_params.max_gap_param);
            printf("New score: %.2f\n", best_score);
            continue;
        }
        
        // 2. Try increasing/decreasing max_diag_diff_param
        test_params = current_params;
        test_params.max_diag_diff_param += step_sizes.max_diag_diff_step;
        apply_constraints(&test_params);
        
        score = evaluate_params(test_params, ref1, query1, ref2, query2);
        if (score > best_score) {
            best_score = score;
            best_params = test_params;
            current_params = test_params;
            improved = true;
            printf("Iteration %d, improved by increasing max_diag_diff_param to %d\n", 
                   iteration + 1, test_params.max_diag_diff_param);
            printf("New score: %.2f\n", best_score);
            continue;
        }
        
        test_params = current_params;
        test_params.max_diag_diff_param -= step_sizes.max_diag_diff_step;
        apply_constraints(&test_params);
        
        score = evaluate_params(test_params, ref1, query1, ref2, query2);
        if (score > best_score) {
            best_score = score;
            best_params = test_params;
            current_params = test_params;
            improved = true;
            printf("Iteration %d, improved by decreasing max_diag_diff_param to %d\n", 
                   iteration + 1, test_params.max_diag_diff_param);
            printf("New score: %.2f\n", best_score);
            continue;
        }
        
        // 3. Try increasing/decreasing overlap_factor_param
        test_params = current_params;
        test_params.overlap_factor_param += step_sizes.overlap_factor_step;
        apply_constraints(&test_params);
        
        score = evaluate_params(test_params, ref1, query1, ref2, query2);
        if (score > best_score) {
            best_score = score;
            best_params = test_params;
            current_params = test_params;
            improved = true;
            printf("Iteration %d, improved by increasing overlap_factor_param to %.1f\n", 
                   iteration + 1, test_params.overlap_factor_param);
            printf("New score: %.2f\n", best_score);
            continue;
        }
        
        test_params = current_params;
        test_params.overlap_factor_param -= step_sizes.overlap_factor_step;
        apply_constraints(&test_params);
        
        score = evaluate_params(test_params, ref1, query1, ref2, query2);
        if (score > best_score) {
            best_score = score;
            best_params = test_params;
            current_params = test_params;
            improved = true;
            printf("Iteration %d, improved by decreasing overlap_factor_param to %.1f\n", 
                   iteration + 1, test_params.overlap_factor_param);
            printf("New score: %.2f\n", best_score);
            continue;
        }
        
        // 4. Try increasing/decreasing min_anchors_param
        test_params = current_params;
        test_params.min_anchors_param += step_sizes.min_anchors_step;
        apply_constraints(&test_params);
        
        score = evaluate_params(test_params, ref1, query1, ref2, query2);
        if (score > best_score) {
            best_score = score;
            best_params = test_params;
            current_params = test_params;
            improved = true;
            printf("Iteration %d, improved by increasing min_anchors_param to %d\n", 
                   iteration + 1, test_params.min_anchors_param);
            printf("New score: %.2f\n", best_score);
            continue;
        }
        
        test_params = current_params;
        test_params.min_anchors_param -= step_sizes.min_anchors_step;
        apply_constraints(&test_params);
        
        score = evaluate_params(test_params, ref1, query1, ref2, query2);
        if (score > best_score) {
            best_score = score;
            best_params = test_params;
            current_params = test_params;
            improved = true;
            printf("Iteration %d, improved by decreasing min_anchors_param to %d\n", 
                   iteration + 1, test_params.min_anchors_param);
            printf("New score: %.2f\n", best_score);
            continue;
        }
        
        // Check if we improved
        if (!improved) {
            no_improvement_count++;
            printf("Iteration %d, no improvement (%d/%d)\n", 
                   iteration + 1, no_improvement_count, max_no_improvement);
            
            // Reduce step sizes
            if (no_improvement_count % 2 == 0) {
                step_sizes.max_gap_step = fmax(step_sizes.max_gap_step * 0.5, 1);
                step_sizes.max_diag_diff_step = fmax(step_sizes.max_diag_diff_step * 0.5, 1);
                step_sizes.overlap_factor_step = fmax(step_sizes.overlap_factor_step * 0.5, 0.05);
                step_sizes.min_anchors_step = fmax(step_sizes.min_anchors_step * 0.5, 1);
                
                printf("Reduced step sizes: max_gap_step=%d, max_diag_diff_step=%d, overlap_factor_step=%.2f, min_anchors_step=%d\n",
                       step_sizes.max_gap_step, step_sizes.max_diag_diff_step,
                       step_sizes.overlap_factor_step, step_sizes.min_anchors_step);
            }
            
            if (no_improvement_count >= max_no_improvement) {
                printf("Stopping: No improvement for %d iterations\n", max_no_improvement);
                break;
            }
        } else {
            no_improvement_count = 0;
        }
    }
    
    // Print best results
    printf("\nParameter tuning finished.\n");
    printf("Best score: %.2f\n", best_score);
    printf("Best parameters found: max_gap=%d, max_diag_diff=%d, overlap_factor=%.1f, min_anchors=%d\n",
           best_params.max_gap_param, best_params.max_diag_diff_param,
           best_params.overlap_factor_param, best_params.min_anchors_param);
}

int main() {
    printf("Parameter tuning for sequence alignment algorithm\n\n");
    
    // These would be loaded from actual sequence files
    const char* ref1 = "ACGTACGTACGTACGT";
    const char* query1 = "ACGTACGTACGT";
    const char* ref2 = "GCATGCATGCATGCAT";
    const char* query2 = "GCATGCAT";
    
    tune_parameters(ref1, query1, ref2, query2);
    
    return 0;
}
