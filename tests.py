from dna_aligner import DNAAligner, reverse_complement
import time
import random

def test_basic_alignment():
    """测试基本对齐功能"""
    print("测试基本对齐...")
    ref = "ATGGTACGATTC"
    query = "ATGCGAGTATTC"
    
    aligner = DNAAligner(k=3, min_anchor_len=3, max_freq=10)
    result = aligner.align(ref, query)
    
    print("参考序列:", ref)
    print("查询序列:", query)
    print("比对结果:", result)
    print()

def test_large_sequences():
    """测试大序列的性能"""
    print("测试大序列性能...")
    
    # 生成随机序列
    bases = ['A', 'T', 'G', 'C']
    ref_len = 10000
    query_len = 9000
    
    ref = ''.join(random.choice(bases) for _ in range(ref_len))
    query = ''.join(random.choice(bases) for _ in range(query_len))
    
    # 插入一些相同区域以确保有匹配
    shared_regions = 5
    for i in range(shared_regions):
        shared_seq = ''.join(random.choice(bases) for _ in range(500))
        ref_pos = random.randint(0, ref_len - 500)
        query_pos = random.randint(0, query_len - 500)
        ref = ref[:ref_pos] + shared_seq + ref[ref_pos+500:]
        query = query[:query_pos] + shared_seq + query[query_pos+500:]
    
    print(f"参考序列长度: {len(ref)}")
    print(f"查询序列长度: {len(query)}")
    
    start_time = time.time()
    aligner = DNAAligner(k=15, min_anchor_len=30, max_freq=10)
    result = aligner.align(ref, query)
    end_time = time.time()
    
    print(f"找到 {len(result)} 个匹配区域")
    print(f"处理时间: {end_time - start_time:.2f} 秒")
    print()

def test_reverse_complement():
    """测试反向互补序列的比对"""
    print("测试反向互补序列...")
    ref = "ATGCTAGCTAGCTA"
    query = reverse_complement("TAGCTAGCTAGCAT")  # ref的反向互补
    
    print("参考序列:", ref)
    print("查询序列:", query)
    print("查询序列是参考序列的反向互补")
    
    # 此测试需要修改DNAAligner类以支持反向互补匹配
    # 这里仅作为示例
    print("注意：当前版本不直接支持反向互补匹配，需要扩展实现")
    print()

def test_mutations():
    """测试各种突变类型"""
    print("测试各种突变类型...")
    
    # 原始序列
    original = "ATGCTAGCTAGCTATCGATCGATCGTAGCTA"
    
    # 插入突变
    insertion = "ATGCTAGCTAGCTAAAAATCGATCGATCGTAGCTA"
    
    # 删除突变
    deletion = "ATGCTAGCTAGCTTCGATCGTAGCTA"
    
    # SNP突变
    snp = "ATGCTAGCTCGCTATCGATCGATCGTAGCTA"
    
    # 测试每种突变
    aligner = DNAAligner(k=5, min_anchor_len=5, max_freq=10)
    
    print("测试插入突变:")
    result = aligner.align(original, insertion)
    print(result)
    
    print("\n测试删除突变:")
    result = aligner.align(original, deletion)
    print(result)
    
    print("\n测试SNP突变:")
    result = aligner.align(original, snp)
    print(result)

if __name__ == "__main__":
    test_basic_alignment()
    test_mutations()
    # 大序列测试可能需要较长时间
    test_large_sequences()
    test_reverse_complement()
