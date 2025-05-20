#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <tuple>

// Function to get reverse complement of a DNA sequence
std::string get_rc(const std::string& s) {
    std::unordered_map<char, char> map_dict = {{'A', 'T'}, {'T', 'A'}, {'G', 'C'}, {'C', 'G'}, {'N', 'N'}};
    std::string result;
    result.reserve(s.length());
    
    for (auto it = s.rbegin(); it != s.rend(); ++it) {
        result.push_back(map_dict[*it]);
    }
    
    return result;
}

// Alias for get_rc for consistency with Python code
std::string rc(const std::string& s) {
    return get_rc(s);
}

// Custom hashing function similar to Python's hash for strings
size_t hash_string(const std::string& s) {
    std::hash<std::string> hasher;
    return hasher(s);
}

// Equivalent to Python's seq2hashtable_multi_test
std::vector<std::tuple<int, int, int, int>> seq2hashtable_multi_test(
    const std::string& refseq, const std::string& testseq, int kmersize = 15, int shift = 1) {
    
    std::string rc_testseq = get_rc(testseq);
    int testseq_len = testseq.length();
    std::unordered_map<size_t, std::vector<int>> local_lookuptable;
    
    size_t skiphash = hash_string(std::string(kmersize, 'N'));
    
    // Build lookup table from reference sequence - Fix for signed/unsigned comparison
    for (size_t iloc = 0; iloc <= refseq.length() - kmersize; iloc += 1) {
        std::string kmer = refseq.substr(iloc, kmersize);
        size_t hashedkmer = hash_string(kmer);
        
        if (skiphash == hashedkmer) {
            continue;
        }
        
        local_lookuptable[hashedkmer].push_back(static_cast<int>(iloc));
    }
    
    std::vector<std::tuple<int, int, int, int>> one_mapinfo;
    int iloc = -1;
    int readend = testseq_len - kmersize + 1;
    
    while (true) {
        iloc += shift;
        if (iloc >= readend) {
            break;
        }
        
        // Check forward strand
        std::string kmer = testseq.substr(iloc, kmersize);
        size_t hashedkmer = hash_string(kmer);
        
        if (local_lookuptable.find(hashedkmer) != local_lookuptable.end()) {
            for (int refloc : local_lookuptable[hashedkmer]) {
                one_mapinfo.push_back(std::make_tuple(iloc, refloc, 1, kmersize));
            }
        }
        
        // Check reverse strand
        std::string rc_kmer = rc_testseq.substr(rc_testseq.length() - (iloc + kmersize), kmersize);
        hashedkmer = hash_string(rc_kmer);
        
        if (local_lookuptable.find(hashedkmer) != local_lookuptable.end()) {
            for (int refloc : local_lookuptable[hashedkmer]) {
                one_mapinfo.push_back(std::make_tuple(iloc, refloc, -1, kmersize));
            }
        }
    }
    
    return one_mapinfo;
}

// Parse points from a string
std::vector<int> get_points(const std::string& tuples_str) {
    std::vector<int> data;
    int num = 0;
    
    for (char c : tuples_str) {
        if (c >= '0' && c <= '9') {
            num = num * 10 + (c - '0');
        }
        else if (c == ',') {
            data.push_back(num);
            num = 0;
        }
    }
    
    if (!tuples_str.empty()) {
        data.push_back(num);
    }
    
    return data;
}

// Calculate edit distance between sequences
int calculate_distance(const std::string& ref, const std::string& query, 
                      int ref_st, int ref_en, int query_st, int query_en) {
    std::string A = ref.substr(ref_st, ref_en - ref_st);
    std::string a = query.substr(query_st, query_en - query_st);
    std::string _a = rc(query.substr(query_st, query_en - query_st));
    
    // Simple Levenshtein distance implementation (to replace edlib)
    auto edit_distance = [](const std::string& s1, const std::string& s2) {
        int m = s1.length();
        int n = s2.length();
        std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1));
        
        for (int i = 0; i <= m; i++) {
            for (int j = 0; j <= n; j++) {
                if (i == 0) {
                    dp[i][j] = j;
                }
                else if (j == 0) {
                    dp[i][j] = i;
                }
                else if (s1[i - 1] == s2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1];
                }
                else {
                    dp[i][j] = 1 + std::min({dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]});
                }
            }
        }
        
        return dp[m][n];
    };
    
    return std::min(edit_distance(A, a), edit_distance(A, _a));
}

// Calculate alignment score
int calculate_value(const std::string& tuples_str, const std::string& ref, const std::string& query) {
    std::vector<int> slicepoints = get_points(tuples_str);
    
    if (!slicepoints.empty() && slicepoints.size() % 4 == 0) {
        int editdistance = 0;
        int aligned = 0;
        int preend = 0;
        
        // Create tuples from points
        std::vector<std::tuple<int, int, int, int>> points;
        for (size_t i = 0; i < slicepoints.size(); i += 4) {
            points.push_back(std::make_tuple(
                slicepoints[i], slicepoints[i + 1], slicepoints[i + 2], slicepoints[i + 3]
            ));
        }
        
        // Sort by first element (query_st)
        std::sort(points.begin(), points.end(), 
            [](const auto& a, const auto& b) { return std::get<0>(a) < std::get<0>(b); });
        
        for (const auto& onetuple : points) {
            int query_st = std::get<0>(onetuple);
            int query_en = std::get<1>(onetuple);
            int ref_st = std::get<2>(onetuple);
            int ref_en = std::get<3>(onetuple);
            
            if (preend > query_st) {
                return 0;
            }
            
            if (query_en - query_st < 30) {
                continue;
            }
            
            preend = query_en;
            
            double error_rate = calculate_distance(ref, query, ref_st, ref_en, query_st, query_en) / 
                                static_cast<double>(query_en - query_st);
            
            if (error_rate > 0.1) {
                continue;
            }
            
            editdistance += calculate_distance(ref, query, ref_st, ref_en, query_st, query_en);
            aligned += query_en - query_st;
        }
        
        return std::max(aligned - editdistance, 0);
    }
    else {
        return 0;
    }
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
            points_list[i], points_list[i + 1], points_list[i + 2], points_list[i + 3]
        ));
    }
    
    return formatted_list;
}

// Read sequence from file
std::string read_sequence_from_file(const std::string& filename) {
    std::ifstream file(filename);
    std::string sequence;
    std::string line;
    
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return "";
    }
    
    while (std::getline(file, line)) {
        sequence += line;
    }
    
    file.close();
    return sequence;
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

// Add a wrapper main function that calls main_eval
// This main will be used when compiling eval.cpp alone
#ifndef PARAMETER_TUNING
int main() {
    return main_eval();
}
#endif
