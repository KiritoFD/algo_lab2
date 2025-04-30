from collections import defaultdict
import networkx as nx

class DNAAligner:
    def __init__(self, k=15, min_anchor_len=20, max_kmer_freq=10, lambda_gap=2):
        """
        初始化DNA序列比对器
        
        参数:
            k: k-mer的长度
            min_anchor_len: 最小锚点长度
            max_kmer_freq: k-mer最大出现频率阈值
            lambda_gap: 间隙惩罚因子
        """
        self.k = k
        self.min_anchor_len = min_anchor_len
        self.max_kmer_freq = max_kmer_freq
        self.lambda_gap = lambda_gap
    
    def generate_kmers(self, sequence):
        """生成序列的k-mers及其位置"""
        kmers = []
        for i in range(len(sequence) - self.k + 1):
            kmer = sequence[i:i+self.k]
            kmers.append((kmer, i))
        return kmers
    
    def build_hash_table(self, ref_kmers):
        """构建参考序列k-mer的哈希表"""
        hash_table = defaultdict(list)
        for kmer, pos in ref_kmers:
            hash_table[kmer].append(pos)
        return hash_table
    
    def reverse_complement(self, seq):
        """生成序列的反向互补序列"""
        complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
        return ''.join([complement.get(base, base) for base in reversed(seq)])
    
    def find_candidate_anchors(self, query_kmers, hash_table, include_reverse=False):
        """查找候选锚点并合并连续匹配"""
        candidate_positions = []
        
        for kmer, q_pos in query_kmers:
            # 检查正向匹配
            if kmer in hash_table and len(hash_table[kmer]) <= self.max_kmer_freq:
                for r_pos in hash_table[kmer]:
                    candidate_positions.append((q_pos, r_pos, False))  # False表示非反向互补
            
            # 检查反向互补匹配
            if include_reverse:
                rc_kmer = self.reverse_complement(kmer)
                if rc_kmer in hash_table and len(hash_table[rc_kmer]) <= self.max_kmer_freq:
                    for r_pos in hash_table[rc_kmer]:
                        candidate_positions.append((q_pos, r_pos, True))  # True表示反向互补
        
        # 按查询位置排序
        candidate_positions.sort(key=lambda x: (x[0], x[1], x[2]))
        
        # 合并连续重叠的k-mers生成锚点
        anchors = []
        current = None
        
        for q_pos, r_pos, is_rc in candidate_positions:
            if not current or current[4] != is_rc:  # 新开始一个锚点或互补性不同
                if current and (current[1] - current[0] >= self.min_anchor_len):
                    anchors.append(tuple(current))
                current = [q_pos, q_pos + self.k, r_pos, r_pos + self.k, is_rc]
            else:
                # 检查是否为连续匹配
                if q_pos == current[1] and r_pos == current[3]:
                    current[1] += 1
                    current[3] += 1
                else:
                    if current[1] - current[0] >= self.min_anchor_len:
                        anchors.append(tuple(current))
                    current = [q_pos, q_pos + self.k, r_pos, r_pos + self.k, is_rc]
        
        # 处理最后一个锚点
        if current and (current[1] - current[0] >= self.min_anchor_len):
            anchors.append(tuple(current))
            
        return anchors
    
    def build_graph(self, anchors):
        """构建锚点之间的有向图"""
        G = nx.DiGraph()
        
        # 添加虚拟起点和终点
        G.add_node("S")
        G.add_node("T")
        
        # 按参考序列起始位置排序锚点
        anchors.sort(key=lambda x: (x[2], x[0]))
        
        # 为每个锚点添加节点和边
        for i, anchor in enumerate(anchors):
            q_start, q_end, r_start, r_end, is_rc = anchor
            score = q_end - q_start  # 锚点基础得分（匹配长度）
            G.add_node(anchor, score=score)
            
            # 连接虚拟起点
            G.add_edge("S", anchor, weight=score)
            
            # 寻找后续锚点
            for j in range(i+1, len(anchors)):
                next_anchor = anchors[j]
                next_q_start, next_q_end, next_r_start, next_r_end, next_is_rc = next_anchor
                
                # 仅连接同向锚点（都是正向或都是反向互补）
                if is_rc == next_is_rc:
                    # 保证参考和查询序列都是在后面的位置
                    if next_r_start > r_end and next_q_start > q_end:
                        # 计算间隙惩罚
                        gap_penalty = self.lambda_gap * (abs(next_r_start - r_end) + abs(next_q_start - q_end))
                        edge_weight = next_q_end - next_q_start - gap_penalty
                        G.add_edge(anchor, next_anchor, weight=edge_weight)
        
        # 连接虚拟终点
        for anchor in anchors:
            G.add_edge(anchor, "T", weight=0)
            
        return G
    
    def longest_path_dp(self, G):
        """使用动态规划找到有向无环图中的最长路径"""
        # 拓扑排序
        topo_order = list(nx.topological_sort(G))
        
        # 初始化DP表和前驱节点
        dp = {node: 0 for node in topo_order}
        prev = {node: None for node in topo_order}
        
        # 动态规划计算最长路径
        for u in topo_order:
            for v in G.successors(u):
                new_score = dp[u] + G[u][v]['weight']
                if new_score > dp[v]:
                    dp[v] = new_score
                    prev[v] = u
        
        # 回溯构建最优路径
        path = []
        current = "T"
        while current != "S":
            path.append(current)
            current = prev[current]
            if current is None:  # 防止异常
                break
        
        path.append("S")
        path.reverse()
        return path
    
    def refine_alignment(self, path, ref, query):
        """对锚点间区域进行局部精调"""
        refined = []
        
        for i in range(1, len(path)-1):  # 跳过虚拟起点终点
            curr = path[i]
            if isinstance(curr, tuple) and len(curr) == 5:  # 确保是有效锚点
                refined.append(curr[:4])  # 只保留位置信息，去掉is_rc标志
        
        return refined
    
    def custom_local_align(self, q_region, r_region):
        """简化版局部序列比对（可以扩展为Smith-Waterman算法）"""
        # 此处可以实现更复杂的局部比对算法
        # 当前仅作为占位符
        return []
    
    def output_format(self, refined_anchors):
        """将精调后的锚点格式化为最终输出"""
        result = []
        for anchor in refined_anchors:
            if isinstance(anchor, tuple) and len(anchor) >= 4:
                q_start, q_end, r_start, r_end = anchor[:4]
                result.append((q_start, q_end, r_start, r_end))
        return sorted(result, key=lambda x: x[0])
    
    def align(self, ref, query, include_reverse=False):
        """主要比对流程"""
        # 步骤1: K-mer哈希与锚点生成
        ref_kmers = self.generate_kmers(ref)
        query_kmers = self.generate_kmers(query)
        hash_table = self.build_hash_table(ref_kmers)
        anchors = self.find_candidate_anchors(query_kmers, hash_table, include_reverse)
        
        if not anchors:
            return []  # 没有找到有效匹配
        
        # 步骤2: 构建图模型
        G = self.build_graph(anchors)
        
        # 步骤3: 动态规划找最长路径
        best_path = self.longest_path_dp(G)
        
        # 步骤4: 局部精调与输出
        refined = self.refine_alignment(best_path, ref, query)
        return self.output_format(refined)
