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

// 前向声明外部函数以调用eval.cpp和run.cpp的功能
std::string read_sequence_from_file(const std::string& filename);
std::vector<std::tuple<int, int, int, int>> seq2hashtable_multi_test(
    const std::string& refseq, const std::string& testseq, int kmersize = 15, int shift = 1);
std::string function(const std::vector<std::tuple<int, int, int, int>>& data,
                     int max_gap_param,
                     int max_diag_diff_param,
                     double overlap_factor_param,
                     int min_anchors_param);
int calculate_value(const std::string& tuples_str, const std::string& ref, const std::string& query);

// 参数优化器类 - 分阶段优化
class TwoStageOptimizer {
private:
    // 参数集合
    struct Parameters {
        int max_gap;
        int max_diag_diff;
        double overlap_factor;
        int min_anchors;
        
        // 用于打印参数
        std::string toString() const {
            std::stringstream ss;
            ss << "gap=" << max_gap << ", diag=" << max_diag_diff 
               << ", overlap=" << overlap_factor << ", anchors=" << min_anchors;
            return ss.str();
        }
    };

    // 数据集
    std::string ref1, que1;
    std::string ref2, que2;
    std::vector<std::tuple<int, int, int, int>> data1, data2;
    
    // 随机数生成器
    std::mt19937 rng;
    
    // 参数取值范围
    const int MIN_GAP = 100, MAX_GAP = 400;
    const int MIN_DIAG = 50, MAX_DIAG = 250;
    const double MIN_OVERLAP = 0.2, MAX_OVERLAP = 0.8;
    const int MIN_ANCHORS = 1, MAX_ANCHORS = 5;
    
    // 记录日志的文件
    std::ofstream log_file;

    // 评估单个数据集上的参数性能
    int evaluate_on_dataset(const Parameters& params, 
                           const std::vector<std::tuple<int, int, int, int>>& data,
                           const std::string& ref, const std::string& query) {
        
        std::string tuples_str = function(data, 
                                        params.max_gap, 
                                        params.max_diag_diff, 
                                        params.overlap_factor,
                                        params.min_anchors);
        
        int score = calculate_value(tuples_str, ref, query);
        return score;
    }

    // 生成邻近参数 (针对模拟退火)
    Parameters generate_neighbor(const Parameters& current, double temperature) {
        Parameters neighbor = current;
        
        // 根据温度调整扰动大小
        int gap_perturbation = static_cast<int>(temperature * 50);
        int diag_perturbation = static_cast<int>(temperature * 30);
        double overlap_perturbation = temperature * 0.1;
        int anchor_perturbation = static_cast<int>(temperature * 1);
        
        // 随机选择一个参数进行扰动
        std::uniform_int_distribution<int> param_choice(0, 3);
        int choice = param_choice(rng);
        
        std::uniform_real_distribution<double> real_dist(-1.0, 1.0);
        std::uniform_int_distribution<int> gap_dist(-gap_perturbation, gap_perturbation);
        std::uniform_int_distribution<int> diag_dist(-diag_perturbation, diag_perturbation);
        std::uniform_int_distribution<int> anchor_dist(-anchor_perturbation, anchor_perturbation);
        
        switch(choice) {
            case 0:
                neighbor.max_gap += gap_dist(rng);
                neighbor.max_gap = std::max(MIN_GAP, std::min(MAX_GAP, neighbor.max_gap));
                break;
            case 1:
                neighbor.max_diag_diff += diag_dist(rng);
                neighbor.max_diag_diff = std::max(MIN_DIAG, std::min(MAX_DIAG, neighbor.max_diag_diff));
                break;
            case 2:
                neighbor.overlap_factor += real_dist(rng) * overlap_perturbation;
                neighbor.overlap_factor = std::max(MIN_OVERLAP, std::min(MAX_OVERLAP, neighbor.overlap_factor));
                break;
            case 3:
                neighbor.min_anchors += anchor_dist(rng);
                neighbor.min_anchors = std::max(MIN_ANCHORS, std::min(MAX_ANCHORS, neighbor.min_anchors));
                break;
        }
        
        return neighbor;
    }

    // 随机生成初始参数
    Parameters generate_random_parameters() {
        std::uniform_int_distribution<int> gap_dist(MIN_GAP, MAX_GAP);
        std::uniform_int_distribution<int> diag_dist(MIN_DIAG, MAX_DIAG);
        std::uniform_real_distribution<double> overlap_dist(MIN_OVERLAP, MAX_OVERLAP);
        std::uniform_int_distribution<int> anchor_dist(MIN_ANCHORS, MAX_ANCHORS);
        
        Parameters params;
        params.max_gap = gap_dist(rng);
        params.max_diag_diff = diag_dist(rng);
        params.overlap_factor = overlap_dist(rng);
        params.min_anchors = anchor_dist(rng);
        
        return params;
    }

public:
    TwoStageOptimizer(const std::string& ref1_file, const std::string& que1_file,
                     const std::string& ref2_file, const std::string& que2_file)
        : rng(std::random_device{}()) {
        
        // 打开日志文件
        log_file.open("parameter_tuning_log.txt");
        log_file << "Stage,Iteration,Temperature,Gap,DiagDiff,OverlapFactor,MinAnchors,Score,BestScore\n";
        
        // 加载序列数据
        std::cout << "Loading sequence data..." << std::endl;
        ref1 = read_sequence_from_file(ref1_file);
        que1 = read_sequence_from_file(que1_file);
        ref2 = read_sequence_from_file(ref2_file);
        que2 = read_sequence_from_file(que2_file);
        
        // 预处理数据
        std::cout << "Preprocessing data..." << std::endl;
        data1 = seq2hashtable_multi_test(ref1, que1, 9, 1);
        data2 = seq2hashtable_multi_test(ref2, que2, 9, 1);
        
        std::cout << "Initialization complete. Dataset1: " << data1.size() << " k-mer matches, Dataset2: " 
                  << data2.size() << " k-mer matches" << std::endl;
    }
    
    ~TwoStageOptimizer() {
        if (log_file.is_open()) {
            log_file.close();
        }
    }
    
    // 优化两个数据集，分别返回各自的最佳参数
    std::pair<Parameters, Parameters> optimize_separate_datasets() {
        std::cout << "Starting separate optimization for both datasets..." << std::endl;
        
        // 对 dataset1 优化
        std::cout << "\n---- Optimizing for Dataset 1 ----" << std::endl;
        Parameters dataset1_best = simulated_annealing_on_dataset(data1, ref1, que1, "Dataset1", 150);
        
        std::cout << "\nBest parameters for Dataset 1: " << dataset1_best.toString() << std::endl;
        int score1 = evaluate_on_dataset(dataset1_best, data1, ref1, que1);
        std::cout << "Score on Dataset 1: " << score1 << std::endl;
        
        // 独立优化 dataset2
        std::cout << "\n---- Optimizing for Dataset 2 ----" << std::endl;
        Parameters dataset2_best = simulated_annealing_on_dataset(data2, ref2, que2, "Dataset2", 150);
        
        std::cout << "\nBest parameters for Dataset 2: " << dataset2_best.toString() << std::endl;
        int score2 = evaluate_on_dataset(dataset2_best, data2, ref2, que2);
        std::cout << "Score on Dataset 2: " << score2 << std::endl;
        
        // 交叉评估
        int dataset1_on_dataset2 = evaluate_on_dataset(dataset1_best, data2, ref2, que2);
        int dataset2_on_dataset1 = evaluate_on_dataset(dataset2_best, data1, ref1, que1);
        
        std::cout << "\n---- Cross Evaluation ----" << std::endl;
        std::cout << "Dataset1 parameters on Dataset2: " << dataset1_on_dataset2 << std::endl;
        std::cout << "Dataset2 parameters on Dataset1: " << dataset2_on_dataset1 << std::endl;
        
        // 单独输出每个数据集的最佳序列情况
        std::cout << "\n---- Dataset 1 Best Alignment ----" << std::endl;
        std::string tuples_str1 = function(data1, 
                                         dataset1_best.max_gap, 
                                         dataset1_best.max_diag_diff, 
                                         dataset1_best.overlap_factor,
                                         dataset1_best.min_anchors);
        display_alignment(tuples_str1, "Dataset1");
        
        std::cout << "\n---- Dataset 2 Best Alignment ----" << std::endl;
        std::string tuples_str2 = function(data2, 
                                         dataset2_best.max_gap, 
                                         dataset2_best.max_diag_diff, 
                                         dataset2_best.overlap_factor,
                                         dataset2_best.min_anchors);
        display_alignment(tuples_str2, "Dataset2");
        
        // 返回两个数据集各自的最佳参数
        return {dataset1_best, dataset2_best};
    }
    
    // 两阶段模拟退火优化 - 保留此方法以兼容
    Parameters two_stage_simulated_annealing() {
        std::cout << "Starting two-stage simulated annealing optimization..." << std::endl;
        
        // 第一阶段：对 dataset1 优化
        std::cout << "\n---- Stage 1: Optimizing for Dataset 1 ----" << std::endl;
        Parameters stage1_best = simulated_annealing_on_dataset(data1, ref1, que1, "Dataset1", 150);
        
        std::cout << "\nBest parameters for Dataset 1: " << stage1_best.toString() << std::endl;
        int score1 = evaluate_on_dataset(stage1_best, data1, ref1, que1);
        std::cout << "Score on Dataset 1: " << score1 << std::endl;
        
        // 第二阶段：基于第一阶段的结果，对 dataset2 进行微调
        std::cout << "\n---- Stage 2: Fine-tuning for Dataset 2 ----" << std::endl;
        Parameters stage2_best = simulated_annealing_on_dataset(data2, ref2, que2, "Dataset2", 100, stage1_best);
        
        std::cout << "\nBest parameters for Dataset 2: " << stage2_best.toString() << std::endl;
        int score2 = evaluate_on_dataset(stage2_best, data2, ref2, que2);
        std::cout << "Score on Dataset 2: " << score2 << std::endl;
        
        // 验证两个数据集上的总分数
        int total_score1 = evaluate_on_dataset(stage1_best, data1, ref1, que1) + 
                          evaluate_on_dataset(stage1_best, data2, ref2, que2);
        
        int total_score2 = evaluate_on_dataset(stage2_best, data1, ref1, que1) + 
                          evaluate_on_dataset(stage2_best, data2, ref2, que2);
        
        std::cout << "\n---- Final Evaluation ----" << std::endl;
        std::cout << "Dataset1 optimized parameters total score: " << total_score1 << std::endl;
        std::cout << "Dataset2 fine-tuned parameters total score: " << total_score2 << std::endl;
        
        // 选择总分数更高的参数集
        Parameters final_best = (total_score1 > total_score2) ? stage1_best : stage2_best;
        std::cout << "\nFinal best parameters: " << final_best.toString() << std::endl;
        std::cout << "Total score: " << std::max(total_score1, total_score2) << std::endl;
        
        return final_best;
    }

private:
    // 在单个数据集上进行模拟退火优化
    Parameters simulated_annealing_on_dataset(
        const std::vector<std::tuple<int, int, int, int>>& data,
        const std::string& ref, const std::string& query,
        const std::string& dataset_name,
        int iterations = 1000,
        const Parameters& initial_params = Parameters(),
        double initial_temp = 1.0, 
        double cooling_rate = 0.95) {
        
        // 初始化
        Parameters current = initial_params.max_gap == 0 ? 
                           generate_random_parameters() : initial_params;
        
        // 获取当前参数的对齐结果
        std::string current_tuples_str = function(data, 
                                               current.max_gap, 
                                               current.max_diag_diff, 
                                               current.overlap_factor,
                                               current.min_anchors);
        double current_score = evaluate_on_dataset(current, data, ref, query);
        
        Parameters best = current;
        double best_score = current_score;
        std::string best_tuples_str = current_tuples_str;
        
        double temperature = initial_temp;
        
        std::cout << "Initial parameters: " << current.toString() << std::endl;
        std::cout << "Initial score on " << dataset_name << ": " << current_score << std::endl;
        std::cout << "Initial alignment:" << std::endl;
        display_alignment(current_tuples_str, dataset_name);
        std::cout << "------------------------" << std::endl;
        
        // 创建详细日志文件
        std::ofstream detail_log("parameter_tuning_detail_" + dataset_name + ".txt");
        if (detail_log.is_open()) {
            detail_log << "Iteration,Temperature,Gap,DiagDiff,OverlapFactor,MinAnchors,Score,Alignment\n";
        }
        
        // 迭代优化
        for (int i = 0; i < iterations; ++i) {
            // 生成邻居解
            Parameters neighbor = generate_neighbor(current, temperature);
            
            // 获取邻居的对齐结果
            std::string neighbor_tuples_str = function(data, 
                                                    neighbor.max_gap, 
                                                    neighbor.max_diag_diff, 
                                                    neighbor.overlap_factor,
                                                    neighbor.min_anchors);
            double neighbor_score = evaluate_on_dataset(neighbor, data, ref, query);
            
            // 计算接受概率
            double acceptance_probability = 1.0;
            if (neighbor_score < current_score) {
                acceptance_probability = std::exp((neighbor_score - current_score) / temperature);
            }
            
            // 决定是否接受新解
            bool accepted = false;
            std::uniform_real_distribution<double> random(0.0, 1.0);
            if (neighbor_score > current_score || random(rng) < acceptance_probability) {
                current = neighbor;
                current_score = neighbor_score;
                current_tuples_str = neighbor_tuples_str;
                accepted = true;
                
                // 更新最佳解
                if (current_score > best_score) {
                    best = current;
                    best_score = current_score;
                    best_tuples_str = current_tuples_str;
                    
                    std::cout << "Iteration " << i << ": New best parameters! " << best.toString() << std::endl;
                    std::cout << "New best score: " << best_score << std::endl;
                    std::cout << "New best alignment:" << std::endl;
                    display_alignment(best_tuples_str, dataset_name);
                    std::cout << "------------------------" << std::endl;
                }
            }
            
            // 记录日志
            log_file << dataset_name << "," << i << "," << temperature << "," 
                     << current.max_gap << "," << current.max_diag_diff << "," 
                     << current.overlap_factor << "," << current.min_anchors << "," 
                     << current_score << "," << best_score << "\n";
            
            // 记录详细日志
            if (detail_log.is_open()) {
                detail_log << i << "," << temperature << "," 
                        << current.max_gap << "," << current.max_diag_diff << "," 
                        << current.overlap_factor << "," << current.min_anchors << "," 
                        << current_score << ",\"" << current_tuples_str << "\"\n";
            }
            
            // 降温
            temperature *= cooling_rate;
            
            // 定期输出进度 (更频繁地打印，每5次迭代)
            if (i % 5 == 0) {
                std::cout << "Iteration " << i << ": temp=" << temperature << std::endl;
                std::cout << "Current parameters: " << current.toString() << std::endl;
                std::cout << "Current score=" << current_score << " (best=" << best_score << ")" << std::endl;
                if (accepted) {
                    std::cout << "Accepted new solution. Current alignment:" << std::endl;
                    display_alignment(current_tuples_str, dataset_name);
                } else {
                    std::cout << "Rejected new solution." << std::endl;
                }
                std::cout << "------------------------" << std::endl;
            }
        }
        
        if (detail_log.is_open()) {
            detail_log.close();
        }
        
        // 最终结果
        std::cout << "\nFinal best parameters for " << dataset_name << ": " << best.toString() << std::endl;
        std::cout << "Final best score: " << best_score << std::endl;
        std::cout << "Final best alignment:" << std::endl;
        display_alignment(best_tuples_str, dataset_name);
        
        return best;
    }
    
    // 显示一个比对结果 (增强版本，提供更多信息)
    void display_alignment(const std::string& tuples_str, const std::string& dataset_name) {
        if (tuples_str.empty()) {
            std::cout << "No alignment found for " << dataset_name << std::endl;
            return;
        }
        
        std::vector<int> points;
        std::stringstream ss(tuples_str);
        std::string item;
        
        while (std::getline(ss, item, ',')) {
            points.push_back(std::stoi(item));
        }
        
        if (points.size() % 4 != 0) {
            std::cout << "Invalid alignment format" << std::endl;
            return;
        }
        
        int total_query_length = 0;
        
        for (size_t i = 0; i < points.size(); i += 4) {
            int query_start = points[i];
            int query_end = points[i+1];
            int ref_start = points[i+2];
            int ref_end = points[i+3];
            int query_length = query_end - query_start;
            // int ref_length = ref_end - ref_start; // Removed unused variable
            
            total_query_length += query_length;
            
            std::cout << "Segment " << (i/4 + 1) << ": ";
            std::cout << "Query[" << query_start << "-" << query_end << "] ";
            std::cout << "Ref[" << ref_start << "-" << ref_end << "] ";
            std::cout << "Length: " << query_length;
            
            // 添加段间距和潜在的对角线偏移
            if (i > 0) {
                int prev_query_end = points[i-3]; // i-4+1
                int prev_ref_end = points[i-1];   // i-4+3
                int query_gap = query_start - prev_query_end;
                int ref_gap = ref_start - prev_ref_end;
                int diag_shift = (ref_start - query_start) - (prev_ref_end - prev_query_end);
                
                std::cout << " | QueryGap: " << query_gap;
                std::cout << " | RefGap: " << ref_gap;
                std::cout << " | DiagShift: " << diag_shift;
            }
            
            std::cout << std::endl;
        }
        
        std::cout << "Total aligned query length: " << total_query_length << std::endl;
        std::cout << "Number of segments: " << (points.size() / 4) << std::endl;
    }
};

// 将eval.cpp中的main函数声明为extern以避免冲突
extern int main_eval();

// 主函数
int main() {
    std::cout << "Separate Parameter Tuning for Sequence Alignment" << std::endl;
    std::cout << "Finding optimal parameters for each dataset independently" << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 创建优化器并运行，分别优化两个数据集
    TwoStageOptimizer optimizer("ref1.txt", "que1.txt", "ref2.txt", "que2.txt");
    auto [dataset1_params, dataset2_params] = optimizer.optimize_separate_datasets();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    
    std::cout << "\nOptimization completed in " << duration << " seconds" << std::endl;
    std::cout << "Dataset1 best parameters: " << dataset1_params.toString() << std::endl;
    std::cout << "Dataset2 best parameters: " << dataset2_params.toString() << std::endl;
    std::cout << "Check parameter_tuning_log.txt for detailed optimization process" << std::endl;
    
    return 0;
}
