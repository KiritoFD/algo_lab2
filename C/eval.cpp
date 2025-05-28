#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <tuple>
#include <sstream>
#include <regex>

// Include data files
#include "data.h"

// Function declaration from run.cpp
std::string function(const std::vector<std::tuple<int, int, int, int>>& data);

// Declaration for read_sequence_from_file (assumed to be implemented elsewhere)
std::string read_sequence_from_file(const std::string& filename);

// Get reverse complement of DNA sequence
std::string get_rc(const std::string& s) {
    std::unordered_map<char, char> map_dict = {
        {'A', 'T'}, {'T', 'A'}, {'G', 'C'}, {'C', 'G'}, {'N', 'N'}
    };
    std::string result;
    result.reserve(s.length());
    for (auto it = s.rbegin(); it != s.rend(); ++it) {
        result += map_dict[*it];
    }
    return result;
}

// Generate k-mer matches between reference and test sequences
std::vector<std::tuple<int, int, int, int>> seq2hashtable_multi_test(
    const std::string& refseq, const std::string& testseq, 
    int kmersize = 15, int shift = 1) {
    
    std::string rc_testseq = get_rc(testseq);
    int testseq_len = testseq.length();
    std::unordered_map<std::string, std::vector<int>> local_lookuptable;
    std::string skip_kmer(kmersize, 'N');
    
    // Build lookup table for reference sequence
    for (int iloc = 0; iloc <= (int)refseq.length() - kmersize; iloc++) {
        std::string kmer = refseq.substr(iloc, kmersize);
        if (kmer == skip_kmer) continue;
        local_lookuptable[kmer].push_back(iloc);
    }
    
    std::vector<std::tuple<int, int, int, int>> one_mapinfo;
    int readend = testseq_len - kmersize + 1;
    
    for (int iloc = 0; iloc < readend; iloc += shift) {
        // Forward strand
        std::string kmer = testseq.substr(iloc, kmersize);
        if (local_lookuptable.count(kmer)) {
            for (int refloc : local_lookuptable[kmer]) {
                one_mapinfo.push_back(std::make_tuple(iloc, refloc, 1, kmersize));
            }
        }
        
        // Reverse strand
        int rc_start = testseq_len - iloc - kmersize;
        if (rc_start >= 0) {
            std::string rc_kmer = rc_testseq.substr(rc_start, kmersize);
            if (local_lookuptable.count(rc_kmer)) {
                for (int refloc : local_lookuptable[rc_kmer]) {
                    one_mapinfo.push_back(std::make_tuple(iloc, refloc, -1, kmersize));
                }
            }
        }
    }
    
    return one_mapinfo;
}

// Parse numbers from tuple string
std::vector<int> get_points(const std::string& tuples_str) {
    std::vector<int> data;
    std::string num_str;
    
    for (char c : tuples_str) {
        if (c >= '0' && c <= '9') {
            num_str += c;
        } else if (c == ',' || c == ' ' || c == ')') {
            if (!num_str.empty()) {
                data.push_back(std::stoi(num_str));
                num_str.clear();
            }
        }
    }
    if (!num_str.empty()) {
        data.push_back(std::stoi(num_str));
    }
    
    return data;
}

// Simple edit distance calculation (Levenshtein distance)
int calculate_edit_distance(const std::string& s1, const std::string& s2) {
    int m = s1.length();
    int n = s2.length();
    std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1));
    
    for (int i = 0; i <= m; i++) dp[i][0] = i;
    for (int j = 0; j <= n; j++) dp[0][j] = j;
    
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (s1[i-1] == s2[j-1]) {
                dp[i][j] = dp[i-1][j-1];
            } else {
                dp[i][j] = 1 + std::min({dp[i-1][j], dp[i][j-1], dp[i-1][j-1]});
            }
        }
    }
    
    return dp[m][n];
}

// Calculate distance between reference and query segments
int calculate_distance(const std::string& ref, const std::string& query,
                      int ref_st, int ref_en, int query_st, int query_en) {
    std::string ref_seg = ref.substr(ref_st, ref_en - ref_st);
    std::string query_seg = query.substr(query_st, query_en - query_st);
    std::string query_rc = get_rc(query_seg);
    
    int dist1 = calculate_edit_distance(ref_seg, query_seg);
    int dist2 = calculate_edit_distance(ref_seg, query_rc);
    
    return std::min(dist1, dist2);
}

// Calculate final alignment value
int calculate_value(const std::string& tuples_str, 
                   const std::string& ref, const std::string& query) {
    std::vector<int> slicepoints = get_points(tuples_str);
    
    if (slicepoints.empty() || slicepoints.size() % 4 != 0) {
        return 0;
    }
    
    int editdistance = 0;
    int aligned = 0;
    int preend = 0;
    
    // Convert to tuples and sort by first element
    std::vector<std::tuple<int, int, int, int>> points;
    for (size_t i = 0; i < slicepoints.size(); i += 4) {
        points.push_back(std::make_tuple(
            slicepoints[i], slicepoints[i+1], 
            slicepoints[i+2], slicepoints[i+3]
        ));
    }
    
    std::sort(points.begin(), points.end());
    
    for (const auto& point : points) {
        int query_st = std::get<0>(point);
        int query_en = std::get<1>(point);
        int ref_st = std::get<2>(point);
        int ref_en = std::get<3>(point);
        
        if (preend > query_st) {
            return 0; // Overlapping segments
        }
        
        if (query_en - query_st < 30) {
            continue; // Too short
        }
        
        preend = query_en;
        
        int seg_len = query_en - query_st;
        int dist = calculate_distance(ref, query, ref_st, ref_en, query_st, query_en);
        
        if ((double)dist / seg_len > 0.1) {
            continue; // Too many errors
        }
        
        editdistance += dist;
        aligned += seg_len;
    }
    
    return std::max(aligned - editdistance, 0);
}

// Format tuples for display
std::vector<std::tuple<int, int, int, int>> format_tuples_for_display(const std::string& tuples_str) {
    std::vector<int> points_list = get_points(tuples_str);
    std::vector<std::tuple<int, int, int, int>> formatted_list;
    
    if (points_list.empty() || points_list.size() % 4 != 0) {
        return formatted_list;
    }
    
    for (size_t i = 0; i < points_list.size(); i += 4) {
        formatted_list.push_back(std::make_tuple(
            points_list[i], points_list[i+1], 
            points_list[i+2], points_list[i+3]
        ));
    }
    
    return formatted_list;
}

    std::string ref1 = read_sequence_from_file("ref1.txt");
    std::string que1 = read_sequence_from_file("que1.txt");
    std::string ref2 = read_sequence_from_file("ref2.txt");
    std::string que2 = read_sequence_from_file("que2.txt");
    auto data1 = seq2hashtable_multi_test(ref1, que1, 9, 1);
    std::cout << "Dataset 1 k-mer matches count: " << data1.size() << std::endl;
    
    std::string tuples_str1 = function(data1);
    std::cout << "Alignment Result for Dataset 1:" << std::endl;
    auto formatted_output1 = format_tuples_for_display(tuples_str1);
    
    for (const auto& tuple : formatted_output1) {
        std::cout << "(" << std::get<0>(tuple) << ", " << std::get<1>(tuple) 
                  << ", " << std::get<2>(tuple) << ", " << std::get<3>(tuple) << ")" << std::endl;
    }
    
    int score1 = calculate_value(tuples_str1, ref1, que1);
    std::cout << "Final Score for Dataset 1: " << score1 << std::endl << std::endl;
    
    // Process Dataset 2
    std::cout << "Processing Dataset 2..." << std::endl;
    auto data2 = seq2hashtable_multi_test(ref2, que2, 9, 1);
    std::cout << "Dataset 2 k-mer matches count: " << data2.size() << std::endl;
    
    std::string tuples_str2 = function(data2);
    std::cout << "Alignment Result for Dataset 2:" << std::endl;
    auto formatted_output2 = format_tuples_for_display(tuples_str2);
    
    for (const auto& tuple : formatted_output2) {
        std::cout << "(" << std::get<0>(tuple) << ", " << std::get<1>(tuple) 
                  << ", " << std::get<2>(tuple) << ", " << std::get<3>(tuple) << ")" << std::endl;
    }
    
    int score2 = calculate_value(tuples_str2, ref2, que2);
    std::cout << "Final Score for Dataset 2: " << score2 << std::endl;
    
    return 0;
}
// Function to be implemented in run.cpp
extern std::string function(const std::vector<std::tuple<int, int, int, int>>& data);

// Rename the main function to avoid conflicts with parameter_tuning.cpp
int main_eval() {
    // Load sequences from files
    std::string ref1 = read_sequence_from_file("ref1.txt");
    std::string que1 = read_sequence_from_file("que1.txt");
    std::string ref2 = read_sequence_from_file("ref2.txt");
    std::string que2 = read_sequence_from_file("que2.txt");
    
    // Process Dataset 1
    std::cout << "Processing Dataset 1..." << std::endl;
    auto data1 = seq2hashtable_multi_test(ref1, que1, 9, 1);
    std::cout << "Dataset 1 k-mer matches count: " << data1.size() << std::endl;
    
    std::string tuples_str1 = function(data1);
    std::cout << "Alignment Result for Dataset 1:" << std::endl;
    auto formatted_output1 = format_tuples_for_display(tuples_str1);
    
    for (const auto& tuple : formatted_output1) {
        std::cout << "(" << std::get<0>(tuple) << ", " << std::get<1>(tuple) << ", " 
                 << std::get<2>(tuple) << ", " << std::get<3>(tuple) << ")" << std::endl;
    }
    
    int score1 = calculate_value(tuples_str1, ref1, que1);
    std::cout << "Final Score for Dataset 1: " << score1 << std::endl << std::endl;
    
    // Process Dataset 2
    std::cout << "Processing Dataset 2..." << std::endl;
    auto data2 = seq2hashtable_multi_test(ref2, que2, 9, 1);
    std::cout << "Dataset 2 k-mer matches count: " << data2.size() << std::endl;
    
    std::string tuples_str2 = function(data2);
    std::cout << "Alignment Result for Dataset 2:" << std::endl;
    auto formatted_output2 = format_tuples_for_display(tuples_str2);
    
    for (const auto& tuple : formatted_output2) {
        std::cout << "(" << std::get<0>(tuple) << ", " << std::get<1>(tuple) << ", " 
                 << std::get<2>(tuple) << ", " << std::get<3>(tuple) << ")" << std::endl;
    }
    
    int score2 = calculate_value(tuples_str2, ref2, que2);
    std::cout << "Final Score for Dataset 2: " << score2 << std::endl;
    
    return 0;
}
