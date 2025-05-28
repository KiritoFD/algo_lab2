CXX = g++
CXXFLAGS = -std=c++17 -Wall -O2
PYTHON = python3

# 主要目标
all: eval parameter_tuning gradient_optimizer

# C++ 程序目标
cpp_all: eval parameter_tuning gradient_optimizer

# 基础评估程序
eval: eval.cpp run.cpp
	$(CXX) $(CXXFLAGS) -o eval eval.cpp run.cpp
