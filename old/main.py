from dna_aligner import DNAAligner
import sys
import time

def main():
    """主程序入口"""
    if len(sys.argv) != 3:
        print("Usage: python main.py <query_file> <reference_file>")
        return
    
    query_file = sys.argv[1]
    reference_file = sys.argv[2]
    
    # 读取序列
    with open(query_file, 'r') as f:
        query = f.read().strip()
    
    with open(reference_file, 'r') as f:
        reference = f.read().strip()
    
    print(f"Query长度: {len(query)}, Reference长度: {len(reference)}")
    
    # 创建DNA比对器实例 - 调整参数以获得更好的结果
    aligner = DNAAligner(k=11, min_anchor_length=15, min_chain_score=20, max_gap=2000)
    
    # 执行比对前先检查初始化
    print("检查初始化...")
    aligner.check_initialization()
    
    # 执行比对
    print("开始DNA序列比对...")
    start_time = time.time()
    matches = aligner.align(query, reference)
    end_time = time.time()
    print(f"比对完成，用时: {end_time - start_time:.2f}秒")
    
    # 输出结果
    if matches:
        formatted_tuples = []
        for q_start, q_end, r_start, r_end in matches:
            formatted_tuples.append(f"( {q_start}, {q_end}, {r_start},{r_end})")
        
        output = "[" + ", ".join(formatted_tuples) + "]"
        print(output)
    else:
        print("[]")

if __name__ == "__main__":
    main()
