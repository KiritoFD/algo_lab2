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

    // Parameter ranges and steps
    const int MIN_GAP = 100, MAX_GAP = 400, STEP_GAP = 10;
    const int MIN_DIAG = 50, MAX_DIAG = 250, STEP_DIAG = 10;
    const double MIN_OVERLAP = 0.2, MAX_OVERLAP = 0.8, STEP_OVERLAP = 0.05;
    const int MIN_ANCHORS = 1, MAX_ANCHORS = 5, STEP_ANCHORS = 1;

    std::ofstream log_file;

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
        int max_iterations = 100) {

        std::cout << "\n---- Optimizing for " << dataset_name << " using Gradient Ascent ----" << std::endl;

        Parameters current_params = generate_random_parameters();
        int current_score = evaluate_on_dataset(current_params, current_data, current_ref, current_query);

        std::cout << "Initial Random Params: " << current_params.toString() << " -> Score: " << current_score << std::endl;
        log_file << dataset_name << ",0,Initial," << current_params.max_gap << "," << current_params.max_diag_diff
                 << "," << current_params.overlap_factor << "," << current_params.min_anchors << "," << current_score << "\n";


        for (int iter = 0; iter < max_iterations; ++iter) {
            Parameters best_neighbor_params = current_params;
            int best_neighbor_score = current_score;
            bool improved = false;

            // Explore neighbors by changing one parameter at a time
            for (int param_idx = 0; param_idx < 4; ++param_idx) { // 4 parameters
                for (int direction = -1; direction <= 1; direction += 2) { // -1 and 1
                    Parameters neighbor_params = current_params;
                    std::string change_type = "";

                    if (param_idx == 0) { // max_gap
                        neighbor_params.max_gap += direction * STEP_GAP;
                        neighbor_params.max_gap = std::max(MIN_GAP, std::min(MAX_GAP, neighbor_params.max_gap));
                        change_type = "max_gap";
                    } else if (param_idx == 1) { // max_diag_diff
                        neighbor_params.max_diag_diff += direction * STEP_DIAG;
                        neighbor_params.max_diag_diff = std::max(MIN_DIAG, std::min(MAX_DIAG, neighbor_params.max_diag_diff));
                        change_type = "max_diag_diff";
                    } else if (param_idx == 2) { // overlap_factor
                        neighbor_params.overlap_factor += direction * STEP_OVERLAP;
                        neighbor_params.overlap_factor = std::max(MIN_OVERLAP, std::min(MAX_OVERLAP, neighbor_params.overlap_factor));
                         // Ensure discrete steps for overlap factor if desired, or round
                        neighbor_params.overlap_factor = std::round(neighbor_params.overlap_factor / STEP_OVERLAP) * STEP_OVERLAP;
                        change_type = "overlap_factor";
                    } else { // min_anchors
                        neighbor_params.min_anchors += direction * STEP_ANCHORS;
                        neighbor_params.min_anchors = std::max(MIN_ANCHORS, std::min(MAX_ANCHORS, neighbor_params.min_anchors));
                        change_type = "min_anchors";
                    }

                    // Avoid re-evaluating if parameter didn't change (due to bounds)
                    if (neighbor_params.max_gap == current_params.max_gap &&
                        neighbor_params.max_diag_diff == current_params.max_diag_diff &&
                        std::abs(neighbor_params.overlap_factor - current_params.overlap_factor) < 1e-5 &&
                        neighbor_params.min_anchors == current_params.min_anchors) {
                        continue;
                    }
                    
                    int neighbor_score = evaluate_on_dataset(neighbor_params, current_data, current_ref, current_query);
                    log_file << dataset_name << "," << iter + 1 << ",Explore_" << change_type << (direction > 0 ? "_Inc" : "_Dec") << ","
                             << neighbor_params.max_gap << "," << neighbor_params.max_diag_diff << ","
                             << neighbor_params.overlap_factor << "," << neighbor_params.min_anchors << "," << neighbor_score << "\n";

                    if (neighbor_score > best_neighbor_score) {
                        best_neighbor_score = neighbor_score;
                        best_neighbor_params = neighbor_params;
                        improved = true;
                    }
                }
            }

            if (improved) {
                current_params = best_neighbor_params;
                current_score = best_neighbor_score;
                std::cout << "Iter " << iter + 1 << ": Improved to Score " << current_score
                          << " with Params: " << current_params.toString() << std::endl;
                log_file << dataset_name << "," << iter + 1 << ",Update," << current_params.max_gap << "," << current_params.max_diag_diff
                         << "," << current_params.overlap_factor << "," << current_params.min_anchors << "," << current_score << "\n";
            } else {
                std::cout << "Iter " << iter + 1 << ": No improvement found. Stopping." << std::endl;
                break; // Local optimum reached
            }
        }
        std::cout << "Optimization for " << dataset_name << " finished." << std::endl;
        std::cout << "Best Params: " << current_params.toString() << " -> Score: " << current_score << std::endl;
        
        std::string final_tuples_str = function(current_data, current_params.max_gap, current_params.max_diag_diff, current_params.overlap_factor, current_params.min_anchors);
        display_alignment_details(final_tuples_str, dataset_name);

        return current_params;
    }

    void run_optimization() {
        Parameters best_params_ds1 = optimize_for_dataset("Dataset1", data1, ref1, que1);
        Parameters best_params_ds2 = optimize_for_dataset("Dataset2", data2, ref2, que2);

        std::cout << "\n--- Final Results ---" << std::endl;
        std::cout << "Best parameters for Dataset 1: " << best_params_ds1.toString() << std::endl;
        std::cout << "Score: " << evaluate_on_dataset(best_params_ds1, data1, ref1, que1) << std::endl;

        std::cout << "Best parameters for Dataset 2: " << best_params_ds2.toString() << std::endl;
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
