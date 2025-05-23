#include <iostream>
#include <vector>
#include <tuple>
#include <string>
#include <algorithm>
#include <cmath>
#include <limits>
#include <chrono>
#include <random>
#include <fstream>
#include <sstream>

// Forward declarations from eval.cpp and run.cpp
std::string read_sequence_from_file(const std::string& filename);
std::vector<std::tuple<int, int, int, int>> seq2hashtable_multi_test(
    const std::string& refseq, const std::string& testseq, int kmersize, int shift);
std::string function(const std::vector<std::tuple<int, int, int, int>>& data,
                     int max_gap_param,
                     int max_diag_diff_param,
                     double overlap_factor_param,
                     int min_anchors_param);
int calculate_value(const std::string& tuples_str, const std::string& ref, const std::string& query);

// Parameter structure
struct Parameters {
    int max_gap;
    int max_diag_diff;
    double overlap_factor;
    int min_anchors;

    std::string toString() const {
        std::stringstream ss;
        ss << "gap=" << max_gap << ", diag=" << max_diag_diff
           << ", overlap=" << overlap_factor << ", anchors=" << min_anchors;
        return ss.str();
    }
};

class GradientOptimizer {
private:
    // Datasets
    std::string ref1, que1, ref2, que2;
    std::vector<std::tuple<int, int, int, int>> data1, data2;

    // Random number generator
    std::mt19937 rng;

    // Parameter ranges and base steps
    const int MIN_GAP = 100, MAX_GAP = 400, STEP_GAP = 10;
    const int MIN_DIAG = 50, MAX_DIAG = 250, STEP_DIAG = 10;
    const double MIN_OVERLAP = 0.2, MAX_OVERLAP = 0.8, STEP_OVERLAP = 0.05;
    const int MIN_ANCHORS = 1, MAX_ANCHORS = 5, STEP_ANCHORS = 1;

    std::ofstream log_file;

    // Adaptive step control for each parameter
    struct ParameterControl {
        double step_multiplier = 1.0;
        const double max_multiplier = 3.0; 
        const double min_multiplier = 0.2; 
        const double increase_factor = 1.15; 
        const double decrease_factor = 0.85; 
    };
    std::vector<ParameterControl> param_controls; // Index 0:gap, 1:diag, 2:overlap, 3:anchors


    // Evaluate parameters on a single dataset
    int evaluate_on_dataset(const Parameters& params,
                           const std::vector<std::tuple<int, int, int, int>>& current_data,
                           const std::string& current_ref, const std::string& current_query) {
        std::string tuples_str = function(current_data,
                                        params.max_gap,
                                        params.max_diag_diff,
                                        params.overlap_factor,
                                        params.min_anchors);
        return calculate_value(tuples_str, current_ref, current_query);
    }

    // Generate a random starting point
    Parameters generate_random_parameters() {
        std::uniform_int_distribution<int> gap_dist(MIN_GAP, MAX_GAP);
        std::uniform_int_distribution<int> diag_dist(MIN_DIAG, MAX_DIAG);
        // To get discrete steps for overlap_factor
        int num_overlap_steps = static_cast<int>((MAX_OVERLAP - MIN_OVERLAP) / STEP_OVERLAP);
        std::uniform_int_distribution<int> overlap_step_dist(0, num_overlap_steps);
        std::uniform_int_distribution<int> anchor_dist(MIN_ANCHORS, MAX_ANCHORS);

        Parameters params;
        params.max_gap = gap_dist(rng);
        params.max_diag_diff = diag_dist(rng);
        params.overlap_factor = MIN_OVERLAP + overlap_step_dist(rng) * STEP_OVERLAP;
        params.min_anchors = anchor_dist(rng);
        return params;
    }
    
    void display_alignment_details(const std::string& tuples_str, const std::string& dataset_name) {
        if (tuples_str.empty()) {
            std::cout << "No alignment found for " << dataset_name << std::endl;
            return;
        }
        std::vector<int> points;
        std::stringstream ss(tuples_str);
        std::string item;
        while (std::getline(ss, item, ',')) { points.push_back(std::stoi(item)); }

        if (points.size() % 4 != 0) {
            std::cout << "Invalid alignment format for " << dataset_name << std::endl; return;
        }
        int total_query_length = 0;
        for (size_t i = 0; i < points.size(); i += 4) {
            total_query_length += (points[i+1] - points[i]);
        }
        std::cout << dataset_name << " Alignment: Segments=" << points.size()/4 
                  << ", TotalQueryLength=" << total_query_length << std::endl;
    }


public:
    GradientOptimizer(const std::string& r1_file, const std::string& q1_file,
                      const std::string& r2_file, const std::string& q2_file)
        : rng(std::random_device{}()) {
        param_controls.resize(4); 
        log_file.open("gradient_optimizer_log.txt");
        log_file << "Dataset,Iteration,Type,Gap,DiagDiff,OverlapFactor,MinAnchors,Score\n";

        std::cout << "Loading sequence data..." << std::endl;
        ref1 = read_sequence_from_file(r1_file);
        que1 = read_sequence_from_file(q1_file);
        ref2 = read_sequence_from_file(r2_file);
        que2 = read_sequence_from_file(q2_file);

        std::cout << "Preprocessing data (k-mer hashing)..." << std::endl;
        data1 = seq2hashtable_multi_test(ref1, que1, 9, 1); // Using kmersize=9, shift=1 as example
        data2 = seq2hashtable_multi_test(ref2, que2, 9, 1);
        std::cout << "Initialization complete. Dataset1 matches: " << data1.size()
                  << ", Dataset2 matches: " << data2.size() << std::endl;
    }

    ~GradientOptimizer() {
        if (log_file.is_open()) {
            log_file.close();
        }
    }

    Parameters optimize_for_dataset(
        const std::string& dataset_name,
        const std::vector<std::tuple<int, int, int, int>>& current_data,
        const std::string& current_ref, const std::string& current_query,
        int max_iterations = 500) {

        std::cout << "\n---- Optimizing for " << dataset_name << " using Patient Adaptive Gradient Ascent ----" << std::endl;

        // Reset parameter controls for each new dataset optimization run
        for(auto& pc : param_controls) {
            pc.step_multiplier = 1.0;
        }

        Parameters current_params = generate_random_parameters();
        int current_score = evaluate_on_dataset(current_params, current_data, current_ref, current_query);
        Parameters overall_best_params = current_params;
        int overall_best_score = current_score;

        std::cout << "Initial Random Params: " << current_params.toString() << " -> Score: " << current_score << std::endl;
        log_file << dataset_name << ",0,Initial," << current_params.max_gap << "," << current_params.max_diag_diff
                 << "," << current_params.overlap_factor << "," << current_params.min_anchors << "," << current_score << "\n";

        int no_improvement_streak = 0;
        const int PATIENCE_BEFORE_STEP_DECREASE = 5; // Only start decreasing step sizes after this many stuck iterations
        const int max_no_improvement_streak_for_jump = 30; 


        for (int iter = 0; iter < max_iterations; ++iter) {
            Parameters candidate_next_params = current_params; 
            bool any_param_direction_improved_individually = false;
            bool param_contributed_to_candidate[4] = {false, false, false, false}; 

            // Determine actual step sizes using dynamic multipliers
            int actual_step_gap = std::max(1, static_cast<int>(STEP_GAP * param_controls[0].step_multiplier));
            int actual_step_diag = std::max(1, static_cast<int>(STEP_DIAG * param_controls[1].step_multiplier));
            double actual_step_overlap = std::max(0.01, STEP_OVERLAP * param_controls[2].step_multiplier);
            int actual_step_anchors = std::max(1, static_cast<int>(STEP_ANCHORS * param_controls[3].step_multiplier));

            // --- Assess max_gap (param_idx = 0) ---
            Parameters temp_params_gap_inc = current_params;
            temp_params_gap_inc.max_gap = std::min(MAX_GAP, current_params.max_gap + actual_step_gap);
            int score_gap_inc = evaluate_on_dataset(temp_params_gap_inc, current_data, current_ref, current_query);
            log_file << dataset_name << "," << iter + 1 << ",Explore_max_gap_Inc(M" << param_controls[0].step_multiplier << ")," << temp_params_gap_inc.max_gap << "," << temp_params_gap_inc.max_diag_diff << "," << temp_params_gap_inc.overlap_factor << "," << temp_params_gap_inc.min_anchors << "," << score_gap_inc << "\n";

            Parameters temp_params_gap_dec = current_params;
            temp_params_gap_dec.max_gap = std::max(MIN_GAP, current_params.max_gap - actual_step_gap);
            int score_gap_dec = evaluate_on_dataset(temp_params_gap_dec, current_data, current_ref, current_query);
            log_file << dataset_name << "," << iter + 1 << ",Explore_max_gap_Dec(M" << param_controls[0].step_multiplier << ")," << temp_params_gap_dec.max_gap << "," << temp_params_gap_dec.max_diag_diff << "," << temp_params_gap_dec.overlap_factor << "," << temp_params_gap_dec.min_anchors << "," << score_gap_dec << "\n";

            if (score_gap_inc > current_score && score_gap_inc >= score_gap_dec) {
                candidate_next_params.max_gap = temp_params_gap_inc.max_gap;
                any_param_direction_improved_individually = true;
                param_contributed_to_candidate[0] = (candidate_next_params.max_gap != current_params.max_gap);
            } else if (score_gap_dec > current_score) {
                candidate_next_params.max_gap = temp_params_gap_dec.max_gap;
                any_param_direction_improved_individually = true;
                param_contributed_to_candidate[0] = (candidate_next_params.max_gap != current_params.max_gap);
            }

            // --- Assess max_diag_diff (param_idx = 1) ---
            Parameters temp_params_diag_inc = current_params; 
            temp_params_diag_inc.max_diag_diff = std::min(MAX_DIAG, current_params.max_diag_diff + actual_step_diag);
            int score_diag_inc_val = evaluate_on_dataset(temp_params_diag_inc, current_data, current_ref, current_query); // Renamed to avoid conflict
            log_file << dataset_name << "," << iter + 1 << ",Explore_max_diag_diff_Inc(M" << param_controls[1].step_multiplier << ")," << temp_params_diag_inc.max_gap << "," << temp_params_diag_inc.max_diag_diff << "," << temp_params_diag_inc.overlap_factor << "," << temp_params_diag_inc.min_anchors << "," << score_diag_inc_val << "\n";

            Parameters temp_params_diag_dec = current_params;
            temp_params_diag_dec.max_diag_diff = std::max(MIN_DIAG, current_params.max_diag_diff - actual_step_diag);
            int score_diag_dec_val = evaluate_on_dataset(temp_params_diag_dec, current_data, current_ref, current_query); // Renamed to avoid conflict
            log_file << dataset_name << "," << iter + 1 << ",Explore_max_diag_diff_Dec(M" << param_controls[1].step_multiplier << ")," << temp_params_diag_dec.max_gap << "," << temp_params_diag_dec.max_diag_diff << "," << temp_params_diag_dec.overlap_factor << "," << temp_params_diag_dec.min_anchors << "," << score_diag_dec_val << "\n";
            
            if (score_diag_inc_val > current_score && score_diag_inc_val >= score_diag_dec_val) {
                candidate_next_params.max_diag_diff = temp_params_diag_inc.max_diag_diff;
                any_param_direction_improved_individually = true;
                param_contributed_to_candidate[1] = (candidate_next_params.max_diag_diff != current_params.max_diag_diff);
            } else if (score_diag_dec_val > current_score) {
                candidate_next_params.max_diag_diff = temp_params_diag_dec.max_diag_diff;
                any_param_direction_improved_individually = true;
                param_contributed_to_candidate[1] = (candidate_next_params.max_diag_diff != current_params.max_diag_diff);
            }

            // --- Assess overlap_factor (param_idx = 2) ---
            Parameters temp_params_overlap_inc = current_params;
            temp_params_overlap_inc.overlap_factor = std::min(MAX_OVERLAP, current_params.overlap_factor + actual_step_overlap);
            temp_params_overlap_inc.overlap_factor = std::round(temp_params_overlap_inc.overlap_factor / STEP_OVERLAP) * STEP_OVERLAP; 
            int score_overlap_inc_val = evaluate_on_dataset(temp_params_overlap_inc, current_data, current_ref, current_query); // Renamed
            log_file << dataset_name << "," << iter + 1 << ",Explore_overlap_factor_Inc(M" << param_controls[2].step_multiplier << ")," << temp_params_overlap_inc.max_gap << "," << temp_params_overlap_inc.max_diag_diff << "," << temp_params_overlap_inc.overlap_factor << "," << temp_params_overlap_inc.min_anchors << "," << score_overlap_inc_val << "\n";

            Parameters temp_params_overlap_dec = current_params;
            temp_params_overlap_dec.overlap_factor = std::max(MIN_OVERLAP, current_params.overlap_factor - actual_step_overlap);
            temp_params_overlap_dec.overlap_factor = std::round(temp_params_overlap_dec.overlap_factor / STEP_OVERLAP) * STEP_OVERLAP; 
            int score_overlap_dec_val = evaluate_on_dataset(temp_params_overlap_dec, current_data, current_ref, current_query); // Renamed
            log_file << dataset_name << "," << iter + 1 << ",Explore_overlap_factor_Dec(M" << param_controls[2].step_multiplier << ")," << temp_params_overlap_dec.max_gap << "," << temp_params_overlap_dec.max_diag_diff << "," << temp_params_overlap_dec.overlap_factor << "," << temp_params_overlap_dec.min_anchors << "," << score_overlap_dec_val << "\n";

            if (score_overlap_inc_val > current_score && score_overlap_inc_val >= score_overlap_dec_val) {
                candidate_next_params.overlap_factor = temp_params_overlap_inc.overlap_factor;
                any_param_direction_improved_individually = true;
                param_contributed_to_candidate[2] = (std::abs(candidate_next_params.overlap_factor - current_params.overlap_factor) > 1e-6);
            } else if (score_overlap_dec_val > current_score) {
                candidate_next_params.overlap_factor = temp_params_overlap_dec.overlap_factor;
                any_param_direction_improved_individually = true;
                param_contributed_to_candidate[2] = (std::abs(candidate_next_params.overlap_factor - current_params.overlap_factor) > 1e-6);
            }

            // --- Assess min_anchors (param_idx = 3) ---
            Parameters temp_params_anchors_inc = current_params;
            temp_params_anchors_inc.min_anchors = std::min(MAX_ANCHORS, current_params.min_anchors + actual_step_anchors);
            int score_anchors_inc_val = evaluate_on_dataset(temp_params_anchors_inc, current_data, current_ref, current_query); // Renamed
            log_file << dataset_name << "," << iter + 1 << ",Explore_min_anchors_Inc(M" << param_controls[3].step_multiplier << ")," << temp_params_anchors_inc.max_gap << "," << temp_params_anchors_inc.max_diag_diff << "," << temp_params_anchors_inc.overlap_factor << "," << temp_params_anchors_inc.min_anchors << "," << score_anchors_inc_val << "\n";

            Parameters temp_params_anchors_dec = current_params;
            temp_params_anchors_dec.min_anchors = std::max(MIN_ANCHORS, current_params.min_anchors - actual_step_anchors);
            int score_anchors_dec_val = evaluate_on_dataset(temp_params_anchors_dec, current_data, current_ref, current_query); // Renamed
            log_file << dataset_name << "," << iter + 1 << ",Explore_min_anchors_Dec(M" << param_controls[3].step_multiplier << ")," << temp_params_anchors_dec.max_gap << "," << temp_params_anchors_dec.max_diag_diff << "," << temp_params_anchors_dec.overlap_factor << "," << temp_params_anchors_dec.min_anchors << "," << score_anchors_dec_val << "\n";

            if (score_anchors_inc_val > current_score && score_anchors_inc_val >= score_anchors_dec_val) {
                candidate_next_params.min_anchors = temp_params_anchors_inc.min_anchors;
                any_param_direction_improved_individually = true;
                param_contributed_to_candidate[3] = (candidate_next_params.min_anchors != current_params.min_anchors);
            } else if (score_anchors_dec_val > current_score) {
                candidate_next_params.min_anchors = temp_params_anchors_dec.min_anchors;
                any_param_direction_improved_individually = true;
                param_contributed_to_candidate[3] = (candidate_next_params.min_anchors != current_params.min_anchors);
            }
            
            candidate_next_params.max_gap = std::max(MIN_GAP, std::min(MAX_GAP, candidate_next_params.max_gap));
            candidate_next_params.max_diag_diff = std::max(MIN_DIAG, std::min(MAX_DIAG, candidate_next_params.max_diag_diff));
            candidate_next_params.overlap_factor = std::max(MIN_OVERLAP, std::min(MAX_OVERLAP, candidate_next_params.overlap_factor));
            candidate_next_params.min_anchors = std::max(MIN_ANCHORS, std::min(MAX_ANCHORS, candidate_next_params.min_anchors));


            if (any_param_direction_improved_individually) {
                int new_combined_score = evaluate_on_dataset(candidate_next_params, current_data, current_ref, current_query);
                log_file << dataset_name << "," << iter + 1 << ",TryCombined," << candidate_next_params.max_gap << "," << candidate_next_params.max_diag_diff << "," << candidate_next_params.overlap_factor << "," << candidate_next_params.min_anchors << "," << new_combined_score << "\n";

                if (new_combined_score > current_score) {
                    current_params = candidate_next_params;
                    current_score = new_combined_score;
                    no_improvement_streak = 0; 
                    std::cout << "Iter " << iter + 1 << ": Improved (Combined) to Score " << current_score
                              << " with Params: " << current_params.toString() << std::endl;
                    log_file << dataset_name << "," << iter + 1 << ",Update," << current_params.max_gap << "," << current_params.max_diag_diff
                             << "," << current_params.overlap_factor << "," << current_params.min_anchors << "," << current_score << "\n";

                    if (current_score > overall_best_score) {
                        overall_best_score = current_score;
                        overall_best_params = current_params;
                        std::cout << "Iter " << iter + 1 << ": *** New Overall Best Score: " << overall_best_score 
                                  << " with Params: " << overall_best_params.toString() << " ***" << std::endl;
                    }
                    // Adapt step multipliers: increase for contributing, gently decrease for non-contributing
                    for(int i=0; i<4; ++i) {
                        if(param_contributed_to_candidate[i]) { 
                            param_controls[i].step_multiplier = std::min(param_controls[i].max_multiplier, param_controls[i].step_multiplier * param_controls[i].increase_factor);
                        } else { 
                            // Only decrease if it's not already very small
                            if (param_controls[i].step_multiplier > param_controls[i].min_multiplier * 1.1) // Avoid over-shrinking
                                param_controls[i].step_multiplier = std::max(param_controls[i].min_multiplier, param_controls[i].step_multiplier * 0.95); // Very gentle decrease for non-contributors
                        }
                    }
                } else { // Combined move did not improve
                    no_improvement_streak++;
                    std::cout << "Iter " << iter + 1 << ": Combined changes did not improve. Streak: " << no_improvement_streak << std::endl;
                    if (no_improvement_streak > PATIENCE_BEFORE_STEP_DECREASE) {
                        for(int i=0; i<4; ++i) {
                            param_controls[i].step_multiplier = std::max(param_controls[i].min_multiplier, param_controls[i].step_multiplier * param_controls[i].decrease_factor);
                        }
                    }
                }
            } else { // No individual parameter change improved score
                no_improvement_streak++;
                std::cout << "Iter " << iter + 1 << ": No individual param change improved. Streak: " << no_improvement_streak << std::endl;
                if (no_improvement_streak > PATIENCE_BEFORE_STEP_DECREASE) {
                    for(int i=0; i<4; ++i) {
                        param_controls[i].step_multiplier = std::max(param_controls[i].min_multiplier, param_controls[i].step_multiplier * param_controls[i].decrease_factor);
                    }
                }
            }

            if (no_improvement_streak >= max_no_improvement_streak_for_jump) {
                std::cout << "Iter " << iter + 1 << ": Stuck for " << no_improvement_streak 
                          << " iterations. Performing a random jump." << std::endl;
                current_params = generate_random_parameters();
                current_score = evaluate_on_dataset(current_params, current_data, current_ref, current_query);
                no_improvement_streak = 0; 
                // Reset step multipliers after a jump
                for(int i=0; i<4; ++i) {
                    param_controls[i].step_multiplier = 1.0;
                }
                std::cout << "Iter " << iter + 1 << ": Jumped to new random Params: " << current_params.toString() 
                          << " -> Score: " << current_score << std::endl;
                log_file << dataset_name << "," << iter + 1 << ",Jump," << current_params.max_gap << "," << current_params.max_diag_diff
                         << "," << current_params.overlap_factor << "," << current_params.min_anchors << "," << current_score << "\n";
                
                if (current_score > overall_best_score) {
                    overall_best_score = current_score;
                    overall_best_params = current_params;
                     std::cout << "Iter " << iter + 1 << ": *** New Overall Best Score (post-jump): " << overall_best_score 
                               << " with Params: " << overall_best_params.toString() << " ***" << std::endl;
                }
            }
        }
        std::cout << "Optimization for " << dataset_name << " finished after " << max_iterations << " iterations." << std::endl;
        std::cout << "Overall Best Params for " << dataset_name << ": " << overall_best_params.toString() 
                  << " -> Score: " << overall_best_score << std::endl;
        
        std::string final_tuples_str = function(current_data, overall_best_params.max_gap, overall_best_params.max_diag_diff, overall_best_params.overlap_factor, overall_best_params.min_anchors);
        display_alignment_details(final_tuples_str, dataset_name);

        return overall_best_params;
    }

    void run_optimization() {
        Parameters best_params_ds1 = optimize_for_dataset("Dataset1", data1, ref1, que1);
        Parameters best_params_ds2 = optimize_for_dataset("Dataset2", data2, ref2, que2);

        std::cout << "\n--- Final Results ---" << std::endl;
        std::cout << "Overall Best parameters for Dataset 1: " << best_params_ds1.toString() << std::endl; // Changed to reflect overall best
        std::cout << "Score: " << evaluate_on_dataset(best_params_ds1, data1, ref1, que1) << std::endl;

        std::cout << "Overall Best parameters for Dataset 2: " << best_params_ds2.toString() << std::endl; // Changed to reflect overall best
        std::cout << "Score: " << evaluate_on_dataset(best_params_ds2, data2, ref2, que2) << std::endl;

        // Cross-evaluation
        int score_ds1_params_on_ds2 = evaluate_on_dataset(best_params_ds1, data2, ref2, que2);
        int score_ds2_params_on_ds1 = evaluate_on_dataset(best_params_ds2, data1, ref1, que1);
        std::cout << "\nCross-Evaluation:" << std::endl;
        std::cout << "Dataset1's best params on Dataset2 score: " << score_ds1_params_on_ds2 << std::endl;
        std::cout << "Dataset2's best params on Dataset1 score: " << score_ds2_params_on_ds1 << std::endl;
    }
};

int main() {
    std::cout << "Gradient Ascent Parameter Optimizer" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    GradientOptimizer optimizer("ref1.txt", "que1.txt", "ref2.txt", "que2.txt");
    optimizer.run_optimization();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    std::cout << "\nTotal optimization time: " << duration << " seconds." << std::endl;
    std::cout << "Log saved to gradient_optimizer_log.txt" << std::endl;

    return 0;
}
