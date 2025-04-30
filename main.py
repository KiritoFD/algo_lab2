from dna_aligner import DNAAligner
import argparse

def read_sequence_from_file(file_path):
    """从文件中读取DNA序列"""
    try:
        with open(file_path, 'r') as file:
            # 读取并去除所有空白字符
            sequence = ''.join(file.read().split())
            return sequence
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{file_path}'")
        return None
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='DNA序列比对工具')
    parser.add_argument('-r', '--ref', help='参考序列文件路径')
    parser.add_argument('-q', '--query', help='查询序列文件路径')
    parser.add_argument('-k', type=int, default=3, help='k-mer大小 (默认: 3)')
    parser.add_argument('--min-anchor', type=int, default=3, help='最小锚点长度 (默认: 3)')
    parser.add_argument('--max-freq', type=int, default=10, help='最大频率 (默认: 10)')
    args = parser.parse_args()

    # 默认序列
    ref = "ATGGTACGATTC"
    query = "ATGCGAGTATTC"
    
    # 如果提供了文件路径，则从文件读取序列
    if args.ref:
        file_ref = read_sequence_from_file(args.ref)
        if file_ref:
            ref = file_ref
    
    if args.query:
        file_query = read_sequence_from_file(args.query)
        if file_query:
            query = file_query
    
    print("参考序列:", ref)
    print("查询序列:", query)
    
    # 初始化比对器
    aligner = DNAAligner(k=args.k, min_anchor_len=args.min_anchor, max_freq=args.max_freq)
    
    # 执行比对
    result = aligner.align(ref, query)
    
    # 显示结果
    print("\n比对结果:")
    for i, (q_start, q_end, r_start, r_end) in enumerate(result):
        print(f"匹配 {i+1}: query[{q_start}:{q_end}] -> ref[{r_start}:{r_end}]")
        print(f"  Query: {query[q_start:q_end]}")
        print(f"  Ref  : {ref[r_start:r_end]}")
    
    print("\n以元组格式输出:")
    print(result)

if __name__ == "__main__":
    main()
