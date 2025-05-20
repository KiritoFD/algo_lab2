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

# 参数调优程序
parameter_tuning: parameter_tuning.cpp eval.cpp run.cpp
	$(CXX) $(CXXFLAGS) -DPARAMETER_TUNING -o parameter_tuning parameter_tuning.cpp eval.cpp run.cpp

# 梯度优化程序
gradient_optimizer: gradient_optimizer.cpp eval.cpp run.cpp
	$(CXX) $(CXXFLAGS) -DGRADIENT_OPTIMIZATION -o gradient_optimizer gradient_optimizer.cpp eval.cpp run.cpp

# Python 运行目标
python_run:
	$(PYTHON) eval.py

# 运行参数调优（Python 版本）
python_parameter_tuning:
	$(PYTHON) -c "from run import function; import numpy as np; print('Running parameter tuning...')"

# 运行评估程序
run_eval: eval
	./eval

# 运行参数调优程序
run_parameter_tuning: parameter_tuning
	./parameter_tuning

# 运行梯度优化器
run_gradient: gradient_optimizer
	./gradient_optimizer

# 运行所有测试
test: eval parameter_tuning gradient_optimizer
	@echo "Running basic evaluation..."
	./eval
	@echo "Running parameter tuning..."
	./parameter_tuning
	@echo "Running gradient optimization..."
	./gradient_optimizer

# 文档生成
docs:
	@echo "Generating algorithm documentation..."
	$(PYTHON) -c "import pydoc; pydoc.writedoc('run')" 

# 清理编译文件
clean:
	rm -f eval parameter_tuning gradient_optimizer *.o *.txt

# 清理所有生成的文件，包括日志
clean_all: clean
	rm -f *.log *.html gradient_optimizer_log.txt parameter_tuning_log.txt

# 帮助信息
help:
	@echo "使用方法:"
	@echo "  make             - 构建所有 C++ 程序"
	@echo "  make cpp_all     - 构建所有 C++ 程序（与 make 相同）"
	@echo "  make eval        - 构建基础评估程序"
	@echo "  make parameter_tuning - 构建参数调优程序"
	@echo "  make gradient_optimizer - 构建梯度优化程序"
	@echo "  make python_run  - 运行 Python 评估脚本"
	@echo "  make run_eval    - 构建并运行评估程序"
	@echo "  make run_parameter_tuning - 构建并运行参数调优"
	@echo "  make run_gradient - 构建并运行梯度优化器"
	@echo "  make test        - 运行所有测试"
	@echo "  make docs        - 生成算法文档"
	@echo "  make clean       - 删除编译文件"
	@echo "  make clean_all   - 删除所有生成的文件，包括日志"

.PHONY: all cpp_all python_run python_parameter_tuning run_eval run_parameter_tuning run_gradient test docs clean clean_all help
