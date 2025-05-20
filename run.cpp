#include <string>
#include <vector>
#include <tuple>
#include <algorithm>
#include <unordered_map>
#include <sstream>
#include <limits>

struct Anchor {
    int q_s;
    int q_e;
    int r_s;
    int r_e;
    int strand;
    int id;
    int score;
};

struct Segment {
    int q_s;
    int q_e;
    int r_s;
    int r_e;
    int score;
    int strand;
};

// Chain anchors function
std::pair<std::vector<int>, std::vector<int>> chain_anchors(
    const std::vector<Anchor>& anchors,
    int kmersize,
    int max_gap_between_anchors,
    int max_diagonal_difference,
    int max_allowed_overlap) {
    
    int n_anchors = anchors.size();
    std::vector<int> dp_score(n_anchors, kmersize);
    std::vector<int> parent_idx(n_anchors, -1);

    for (int i = 0; i < n_anchors; i++) {
        const auto& anchor_i = anchors[i];

        for (int j = 0; j < i; j++) {
            const auto& anchor_j = anchors[j];
            
            bool can_link = false;
            if (anchor_i.strand == anchor_j.strand) {
                int query_gap = anchor_i.q_s - anchor_j.q_e;
                
                int diag_j = 0, diag_i = 0;
                int ref_gap = 0;

                if (anchor_i.strand == 1) { // Forward strand
                    ref_gap = anchor_i.r_s - anchor_j.r_e;
                    diag_j = anchor_j.r_s - anchor_j.q_s;
                    diag_i = anchor_i.r_s - anchor_i.q_s;
                } else { // Reverse strand
                    ref_gap = anchor_j.r_s - anchor_i.r_e;
                    diag_j = anchor_j.r_s + anchor_j.q_s;
                    diag_i = anchor_i.r_s + anchor_i.q_s;
                }
                
                if (query_gap >= -max_allowed_overlap && query_gap <= max_gap_between_anchors &&
                    ref_gap >= -max_allowed_overlap && ref_gap <= max_gap_between_anchors) {
                    if (std::abs(diag_i - diag_j) <= max_diagonal_difference) {
                        can_link = true;
                    }
                }
            }
            
            if (can_link) {
                int current_chain_score = dp_score[j] + kmersize;
                if (current_chain_score > dp_score[i]) {
                    dp_score[i] = current_chain_score;
                    parent_idx[i] = j;
                }
            }
        }
    }
    
    return {dp_score, parent_idx};
}

// Select non-overlapping segments
std::vector<int> select_segments(const std::vector<Segment>& segments) {
    int n_segs = segments.size();
    std::vector<int> dp_select_score(n_segs);
    for (int i = 0; i < n_segs; i++) {
        dp_select_score[i] = segments[i].score;
    }
    
    std::vector<int> prev_select_idx(n_segs, -1);

    for (int i = 0; i < n_segs; i++) {
        int seg_i_q_s = segments[i].q_s;
        int seg_i_score = segments[i].score;
        
        for (int j = 0; j < i; j++) {
            int seg_j_q_e = segments[j].q_e;
            if (seg_j_q_e <= seg_i_q_s) { // Non-overlapping
                if (dp_select_score[j] + seg_i_score > dp_select_score[i]) {
                    dp_select_score[i] = dp_select_score[j] + seg_i_score;
                    prev_select_idx[i] = j;
                }
            }
        }
    }
    
    // Find segment with best score
    int best_end_idx = -1;
    int best_total_score = -1;
    
    for (int i = 0; i < n_segs; i++) {
        if (dp_select_score[i] > best_total_score) {
            best_total_score = dp_select_score[i];
            best_end_idx = i;
        }
    }
    
    // Reconstruct path
    std::vector<int> selected_indices;
    int curr_idx = best_end_idx;
    while (curr_idx != -1) {
        selected_indices.push_back(curr_idx);
        curr_idx = prev_select_idx[curr_idx];
    }
    
    // Reverse to get correct order
    std::reverse(selected_indices.begin(), selected_indices.end());
    
    return selected_indices;
}

// The function called from eval.cpp
std::string function(const std::vector<std::tuple<int, int, int, int>>& data,
                    int max_gap_param = 250,
                    int max_diag_diff_param = 150,
                    double overlap_factor_param = 0.5,
                    int min_anchors_param = 1) {
    
    // Check if data is empty
    if (data.empty()) {
        return "";
    }
    
    // Extract kmersize from the first tuple
    int kmersize = std::get<3>(data[0]);
    if (kmersize <= 0) {
        return "";
    }
    
    // Parameters (now passed directly rather than having defaults)
    int MAX_GAP_BETWEEN_ANCHORS = max_gap_param;
    int MAX_DIAGONAL_DIFFERENCE = max_diag_diff_param;
    int MAX_ALLOWED_OVERLAP = static_cast<int>(kmersize * overlap_factor_param);
    int MIN_ANCHORS_PER_CHAIN = min_anchors_param;
    
    // 1. Convert data to anchors
    std::vector<Anchor> py_anchors;
    for (size_t i = 0; i < data.size(); i++) {
        const auto& tuple = data[i];
        int q_s = std::get<0>(tuple);
        int r_s = std::get<1>(tuple);
        int strand_val = std::get<2>(tuple);
        
        Anchor anchor;
        anchor.q_s = q_s;
        anchor.q_e = q_s + kmersize;
        anchor.r_s = r_s;
        anchor.r_e = r_s + kmersize;
        anchor.strand = strand_val;
        anchor.id = static_cast<int>(i);
        
        py_anchors.push_back(anchor);
    }
    
    // 2. Sort anchors by q_s, r_s
    std::sort(py_anchors.begin(), py_anchors.end(), 
              [](const Anchor& a, const Anchor& b) {
                  return std::tie(a.q_s, a.r_s) < std::tie(b.q_s, b.r_s);
              });
    
    // 3. Chain anchors
    auto [dp_score, parent_idx] = chain_anchors(
        py_anchors, kmersize, MAX_GAP_BETWEEN_ANCHORS, 
        MAX_DIAGONAL_DIFFERENCE, MAX_ALLOWED_OVERLAP);
    
    // 4. Form candidate segments from chains
    std::vector<Segment> candidate_segments;
    
    for (size_t i = 0; i < py_anchors.size(); i++) {
        std::vector<int> current_chain_indices;
        int curr = static_cast<int>(i);
        int num_anchors_in_chain = 0;
        
        while (curr != -1) {
            current_chain_indices.push_back(curr);
            num_anchors_in_chain++;
            curr = parent_idx[curr];
        }
        
        std::reverse(current_chain_indices.begin(), current_chain_indices.end());
        
        if (num_anchors_in_chain >= MIN_ANCHORS_PER_CHAIN) {
            int first_anchor_idx = current_chain_indices[0];
            int last_anchor_idx = current_chain_indices.back();
            
            int q_start = py_anchors[first_anchor_idx].q_s;
            int q_end = py_anchors[last_anchor_idx].q_e;
            
            int r_start = std::numeric_limits<int>::max();
            int r_end = std::numeric_limits<int>::min();
            
            for (int idx : current_chain_indices) {
                r_start = std::min(r_start, py_anchors[idx].r_s);
                r_end = std::max(r_end, py_anchors[idx].r_e);
            }
            
            Segment segment;
            segment.q_s = q_start;
            segment.q_e = q_end;
            segment.r_s = r_start;
            segment.r_e = r_end;
            segment.score = dp_score[static_cast<int>(i)];
            segment.strand = py_anchors[i].strand;
            
            candidate_segments.push_back(segment);
        }
    }
    
    // 5. Sort and select non-overlapping segments
    if (candidate_segments.empty()) {
        return "";
    }
    
    std::sort(candidate_segments.begin(), candidate_segments.end(),
              [](const Segment& a, const Segment& b) {
                  if (a.q_s != b.q_s) return a.q_s < b.q_s;
                  if (a.score != b.score) return a.score > b.score;
                  return a.q_e < b.q_e;
              });
    
    std::vector<int> selected_indices = select_segments(candidate_segments);
    
    // 6. Format output
    std::vector<Segment> final_selected_segments;
    for (int idx : selected_indices) {
        final_selected_segments.push_back(candidate_segments[idx]);
    }
    
    std::stringstream output;
    for (size_t i = 0; i < final_selected_segments.size(); i++) {
        const auto& seg = final_selected_segments[i];
        output << seg.q_s << "," << seg.q_e << "," << seg.r_s << "," << seg.r_e;
        if (i < final_selected_segments.size() - 1) {
            output << ",";
        }
    }
    
    return output.str();
}

// 重载函数，提供向后兼容的默认参数版本
std::string function(const std::vector<std::tuple<int, int, int, int>>& data) {
    // 使用默认参数调用完整版本
    return function(data, 250, 150, 0.5, 1);
}
