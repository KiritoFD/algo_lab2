from dna_aligner import DNAAligner
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np

def visualize_alignment(ref, query, anchors):
    """可视化DNA序列比对结果"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 绘制参考序列
    ax.plot([0, len(ref)], [1, 1], 'k-', linewidth=2, label='参考序列')
    
    # 绘制查询序列
    ax.plot([0, len(query)], [0, 0], 'k-', linewidth=2, label='查询序列')
    
    # 绘制锚点匹配
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(anchors))))
    
    for i, (q_start, q_end, r_start, r_end) in enumerate(anchors):
        color = colors[i % len(colors)]
        # 在参考序列上绘制锚点
        ax.plot([r_start, r_end], [1, 1], '-', color=color, linewidth=4)
        # 在查询序列上绘制锚点
        ax.plot([q_start, q_end], [0, 0], '-', color=color, linewidth=4)
        # 绘制连接线
        ax.plot([r_start, q_start], [1, 0], '--', color=color, alpha=0.5)
        ax.plot([r_end, q_end], [1, 0], '--', color=color, alpha=0.5)
    
    # 设置图表属性
    ax.set_xlim(-1, max(len(ref), len(query)) + 1)
    ax.set_ylim(-0.5, 1.5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['查询序列', '参考序列'])
    ax.set_xlabel('序列位置')
    ax.set_title('DNA序列比对结果')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('alignment_result.png')
    plt.show()

def format_alignment_details(ref, query, anchors):
    """格式化比对详情以便于输出"""
    details = []
    
    for i, (q_start, q_end, r_start, r_end) in enumerate(anchors):
        q_segment = query[q_start:q_end]
        r_segment = ref[r_start:r_end]
        match_len = q_end - q_start
        
        details.append(f"锚点 {i+1}:")
        details.append(f"  查询序列位置: {q_start}-{q_end} (长度: {match_len})")
        details.append(f"  参考序列位置: {r_start}-{r_end} (长度: {r_end - r_start})")
        details.append(f"  查询片段: {q_segment}")
        details.append(f"  参考片段: {r_segment}")
        details.append("")
    
    return "\n".join(details)

def main():
    parser = argparse.ArgumentParser(description="DNA序列比对工具")
    parser.add_argument('-r', '--reference', required=True, help='参考序列文件路径')
    parser.add_argument('-q', '--query', required=True, help='查询序列文件路径')
    parser.add_argument('-k', '--kmer', type=int, default=15, help='k-mer长度 (默认: 15)')
    parser.add_argument('-m', '--min-anchor', type=int, default=20, help='最小锚点长度 (默认: 20)')
    parser.add_argument('-v', '--visualize', action='store_true', help='是否可视化比对结果')
    parser.add_argument('--reverse', action='store_true', help='是否包括反向互补匹配')
    
    args = parser.parse_args()
    
    # 读取序列数据
    try:
        with open(args.reference, 'r') as f:
            ref = "".join(line.strip() for line in f if not line.startswith('>'))
        
        with open(args.query, 'r') as f:
            query = "".join(line.strip() for line in f if not line.startswith('>'))
    except FileNotFoundError:
        print("文件路径错误，请检查文件是否存在!")
        return
    
    print(f"参考序列长度: {len(ref)}bp")
    print(f"查询序列长度: {len(query)}bp")
    
    # 创建比对器并执行比对
    aligner = DNAAligner(k=args.kmer, min_anchor_len=args.min_anchor)
    
    start_time = time.time()
    result = aligner.align(ref, query, include_reverse=args.reverse)
    end_time = time.time()
    
    # 输出结果
    print(f"\n找到 {len(result)} 个锚点匹配")
    print(f"比对用时: {end_time - start_time:.3f} 秒")
    
    if result:
        # 计算匹配覆盖率
        total_q_matched = sum(q_end - q_start for q_start, q_end, _, _ in result)
        total_r_matched = sum(r_end - r_start for _, _, r_start, r_end in result)
        
        q_coverage = total_q_matched / len(query) * 100
        r_coverage = total_r_matched / len(ref) * 100
        
        print(f"查询序列匹配覆盖率: {q_coverage:.2f}%")
        print(f"参考序列匹配覆盖率: {r_coverage:.2f}%")
        
        print("\n比对详情:")
        print(format_alignment_details(ref, query, result))
        
        # 可视化
        if args.visualize:
            visualize_alignment(ref, query, result)
    else:
        print("未找到有效匹配")

if __name__ == "__main__":
    main()
