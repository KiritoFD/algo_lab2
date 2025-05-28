#ifndef DATA_H
#define DATA_H

#include <string>
#include <fstream>
#include <sstream>

// Function to read file content into string
inline std::string read_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();
    
    // Remove any trailing newlines or whitespace
    while (!content.empty() && (content.back() == '\n' || content.back() == '\r' || content.back() == ' ')) {
        content.pop_back();
    }
    
    return content;
}

// Dataset sequences loaded from files
inline const std::string& get_ref1() {
    static std::string ref1 = read_file("ref1.txt");
    return ref1;
}

inline const std::string& get_que1() {
    static std::string que1 = read_file("que1.txt");
    return que1;
}

inline const std::string& get_ref2() {
    static std::string ref2 = read_file("ref2.txt");
    return ref2;
}

inline const std::string& get_que2() {
    static std::string que2 = read_file("que2.txt");
    return que2;
}

#endif // DATA_H

