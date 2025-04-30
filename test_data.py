import random
import argparse

def generate_random_dna(length):
    """生成随机DNA序列"""
    return ''.join(random.choice('ACGT') for _ in range(length))

def introduce_mutations(sequence, sub_rate=0.01, ins_rate=0.005, del_rate=0.005, inv_rate=0.001):
    """向序列中引入突变"""
    result = list(sequence)
    length = len(result)
    
    # 替换突变
    for i in range(length):
        if random.random() < sub_rate:
            result[i] = random.choice([base for base in 'ACGT' if base != result[i]])
    
    # 删除突变
    i = 0
    while i < len(result):
        if random.random() < del_rate:
            del result[i]
        else:
            i += 1
    
    # 插入突变
    i = 0
    inserted = 0
    while i < length + inserted:
        if random.random() < ins_rate:
            result.insert(i, random.choice('ACGT'))
            inserted += 1
        i += 1
    
    # 将结果转换回字符串
    sequence = ''.join(result)
    
    # 逆转突变
    if inv_rate > 0:
        segments = []
        start = 0
        while start < len(sequence):
            segment_len = min(random.randint(50, 200), len(sequence) - start)
            segment = sequence[start:start+segment_len]
            
            # 有一定概率逆转片段
            if random.random() < inv_rate * segment_len:
                complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
                segment = ''.join(complement.get(base, base) for base in reversed(segment))
            
            segments.append(segment)
            start += segment_len
        
        sequence = ''.join(segments)
    
    return sequence

def main():
    parser = argparse.ArgumentParser(description="生成DNA测试数据")
    parser.add_argument('-l', '--length', type=int, default=10000, help='参考序列长度')
    parser.add_argument('-s', '--sub-rate', type=float, default=0.01, help='碱基替换率')
    parser.add_argument('-i', '--ins-rate', type=float, default=0.005, help='碱基插入率')
    parser.add_argument('-d', '--del-rate', type=float, default=0.005, help='碱基删除率')
    parser.add_argument('-v', '--inv-rate', type=float, default=0.001, help='序列逆转率')
    parser.add_argument('-o', '--output', default='test', help='输出文件前缀')
    
    args = parser.parse_args()
    
    # 生成参考序列
    ref_seq = generate_random_dna(args.length)
    with open(f"{args.output}_ref.fasta", "w") as f:
        f.write(f">Reference_Sequence\n")
        for i in range(0, len(ref_seq), 70):
            f.write(f"{ref_seq[i:i+70]}\n")
    
    # 生成突变后的查询序列
    query_seq = introduce_mutations(
        ref_seq, 
        sub_rate=args.sub_rate,
        ins_rate=args.ins_rate,
        del_rate=args.del_rate,
        inv_rate=args.inv_rate
    )
    
    with open(f"{args.output}_query.fasta", "w") as f:
        f.write(f">Query_Sequence\n")
        for i in range(0, len(query_seq), 70):
            f.write(f"{query_seq[i:i+70]}\n")
    
    print(f"生成的参考序列长度: {len(ref_seq)}bp")
    print(f"生成的查询序列长度: {len(query_seq)}bp")
    print(f"文件已保存为 {args.output}_ref.fasta 和 {args.output}_query.fasta")

if __name__ == "__main__":
    main()
