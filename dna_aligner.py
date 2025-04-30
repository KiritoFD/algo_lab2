from collections import defaultdict
import networkx as nx

class DNAAligner:
    def __init__(self, k=3, min_anchor_len=3, max_freq=10, lambda_gap=2):
        """
        初始化DNA序列比对器
        
        参数:
        - k: k-mer大小
        - min_anchor_len: 最小锚点长度
        - max_freq: 最大k-mer频率 (过滤超高频重复序列)
        - lambda_gap: 间隙惩罚系数
        """
        self.k = k
        self.min_anchor_len = min_anchor_len
        self.max_freq = max_freq
        self.lambda_gap = lambda_gap
    
    def reverse_complement(self, seq):
        """返回序列的反向互补序列"""
        complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 
                      'a': 't', 't': 'a', 'c': 'g', 'g': 'c'}
        return ''.join(complement.get(base, base) for base in reversed(seq))
    
    def generate_kmers(self, seq, k):
        """返回所有k-mer及其起始位置"""
        kmers = []
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            kmers.append((kmer, i))
        return kmers
    
    def build_hash_table(self, ref_kmers):
        """构建k-mer哈希表"""
        hash_table = defaultdict(list)
        for kmer, pos in ref_kmers:
            hash_table[kmer].append(pos)
        return hash_table
    
    def find_candidate_anchors(self, query, ref, with_reverse=True):
        """查找两个序列之间的候选锚点"""
        # 生成 query 的 k-mers
        q_kmers = self.generate_kmers(query, self.k)
        candidates = []
        
        # 为反向互补匹配准备数据
        if with_reverse:
            rc_query = self.reverse_complement(query)
            rc_q_kmers = self.generate_kmers(rc_query, self.k)
        
        # 构建 reference 的 k-mer 哈希表
        ref_kmers = self.generate_kmers(ref, self.k)
        hash_table = self.build_hash_table(ref_kmers)
        
        # 查找正向匹配
        for kmer, q_pos in q_kmers:
            if kmer in hash_table and len(hash_table[kmer]) <= self.max_freq:
                for r_pos in hash_table[kmer]:
                    candidates.append((q_pos, r_pos, False))  # False表示非反向互补
        
        # 查找反向互补匹配
        if with_reverse:
            for kmer, q_pos in rc_q_kmers:
                if kmer in hash_table and len(hash_table[kmer]) <= self.max_freq:
                    for r_pos in hash_table[kmer]:
                        # 反向互补的锚点使用特殊标记
                        candidates.append((len(query) - q_pos - self.k, r_pos, True))
        
        # 如果没有找到足够的候选锚点，尝试放宽筛选条件
        if len(candidates) < 5 and self.k > 2:
            # 临时降低k值，重新查找
            old_k = self.k
            self.k = max(2, self.k - 1)
            candidates = self.find_candidate_anchors(query, ref, with_reverse)
            self.k = old_k  # 恢复原来的k值
        
        # 合并连续 k-mer 形成锚点
        anchors = []
        # 分别处理正向和反向互补匹配
        for is_reverse in [False, True]:
            current = None
            # 过滤并排序当前方向的候选锚点
            current_candidates = sorted([c for c in candidates if c[2] == is_reverse], key=lambda x: (x[0], x[1]))
            
            for q_pos, r_pos, _ in current_candidates:
                if not current:
                    current = [q_pos, q_pos + self.k, r_pos, r_pos + self.k, is_reverse]
                else:
                    # 放宽合并条件，允许小间隔（最多1个碱基）
                    if abs(q_pos - current[1]) <= 1 and abs(r_pos - current[3]) <= 1 and is_reverse == current[4]:
                        current[1] = max(current[1], q_pos + self.k)
                        current[3] = max(current[3], r_pos + self.k)
                    else:
                        # 降低最小锚点长度门槛
                        min_len = max(self.min_anchor_len, 2)
                        if current[1] - current[0] >= min_len:
                            anchors.append((current[0], current[1], current[2], current[3]))
                        current = [q_pos, q_pos + self.k, r_pos, r_pos + self.k, is_reverse]
            
            # 处理最后一个锚点
            if current and current[1] - current[0] >= self.min_anchor_len:
                anchors.append((current[0], current[1], current[2], current[3]))
        
        return anchors
    
    def build_graph(self, anchors):
        """构建锚点之间的有向无环图"""
        G = nx.DiGraph()
        G.add_node("S")  # 起点
        G.add_node("T")  # 终点
        
        # 按照 query_start 排序
        anchors.sort(key=lambda x: x[0])
        
        # 添加节点与边
        for i, anchor in enumerate(anchors):
            q_start, q_end, r_start, r_end = anchor
            score = q_end - q_start  # 锚点长度作为得分
            G.add_node(anchor, score=score)
            G.add_edge("S", anchor, weight=score)  # 所有锚点都从起点可达
            
            # 连接其他锚点
            for j in range(i+1, len(anchors)):
                next_anchor = anchors[j]
                next_q_start, next_q_end, next_r_start, next_r_end = next_anchor
                
                # 修复：严格要求两个序列都必须递增
                if next_q_start > q_end and next_r_start > r_end:
                    gap_q = next_q_start - q_end
                    gap_r = next_r_start - r_end
                    # 计算间隙惩罚
                    gap_penalty = self.lambda_gap * (gap_q + gap_r)
                    edge_weight = (next_q_end - next_q_start) - gap_penalty
                    G.add_edge(anchor, next_anchor, weight=edge_weight)
            
            # 连接到终点
            G.add_edge(anchor, "T", weight=0)
        
        return G
    
    def longest_path_dp(self, G):
        """使用动态规划找到图中的最长路径"""
        topo_order = list(nx.topological_sort(G))
        dp = {node: -float('inf') for node in topo_order}
        prev = {node: None for node in topo_order}
        dp["S"] = 0  # 起点得分为0
        
        for u in topo_order:
            for v in list(G.successors(u)):  # 创建副本避免迭代时修改
                if v in dp:  # 确保节点在dp表中
                    new_score = dp[u] + G.edges[u, v]['weight']
                    if new_score > dp[v]:
                        dp[v] = new_score
                        prev[v] = u
        
        # 回溯路径
        path = []
        curr = "T"
        while curr != "S":
            if prev[curr] is None:
                # 如果无法回溯到起点，检查是否有其他可能的路径
                if not path:
                    # 如果当前路径为空，寻找得分最高的节点
                    best_node = None
                    best_score = float('-inf')
                    for node in G.nodes():
                        if node not in ("S", "T") and dp[node] > best_score:
                            best_score = dp[node]
                            best_node = node
                    
                    if best_node:
                        return ["S", best_node, "T"]  # 返回一个简单路径
                return []  # 无法回溯到起点
            path.append(curr)
            curr = prev[curr]
        path.reverse()
        
        return path
    
    def smith_waterman(self, a, b):
        """标准Smith-Waterman局部比对算法"""
        n, m = len(a), len(b)
        dp = [[0] * (m+1) for _ in range(n+1)]
        max_score, max_pos = 0, (0, 0)
        
        # 计算DP表
        for i in range(1, n+1):
            for j in range(1, m+1):
                match = 2 if a[i-1] == b[j-1] else -1  # 匹配得分2，不匹配罚分-1
                dp[i][j] = max(
                    0,
                    dp[i-1][j-1] + match,  # 匹配或替换
                    dp[i-1][j] - 1,        # 插入
                    dp[i][j-1] - 1         # 删除
                )
                if dp[i][j] > max_score:
                    max_score = dp[i][j]
                    max_pos = (i, j)
        
        # 回溯
        i, j = max_pos
        a_align, b_align = '', ''
        while i > 0 and j > 0 and dp[i][j] > 0:
            diag = dp[i-1][j-1] + (2 if a[i-1] == b[j-1] else -1)
            top = dp[i-1][j] - 1
            left = dp[i][j-1] - 1
            
            if dp[i][j] == diag:
                a_align = a[i-1] + a_align
                b_align = b[j-1] + b_align
                i -= 1
                j -= 1
            elif dp[i][j] == top:
                a_align = a[i-1] + a_align
                b_align = '-' + b_align
                i -= 1
            else:
                a_align = '-' + a_align
                b_align = b[j-1] + b_align
                j -= 1
                
        return a_align, b_align, max_pos[0] - len(a_align), max_pos[1] - len(b_align)
    
    def refine_alignment(self, path, ref, query):
        """精调比对路径，优化锚点边界"""
        # 实现真正的Smith-Waterman精调
        refined = []
        for i in range(1, len(path)-1):  # 排除S/T节点
            curr = path[i]
            if curr not in ("S", "T"):
                refined.append(curr)
                
                # 检查是否还有下一个锚点需要精调
                if i+1 < len(path)-1:  # 确保不是最后一个有效节点
                    next_node = path[i+1]
                    if next_node not in ("S", "T"):
                        q_start, q_end, r_start, r_end = curr
                        next_q_start, next_q_end, next_r_start, next_r_end = next_node
                        
                        # 提取间隙区域进行局部比对
                        q_gap = query[q_end:next_q_start]
                        r_gap = ref[r_end:next_r_start]
                        
                        # 只有当间隙区域都存在时才进行局部比对
                        if q_gap and r_gap:
                            try:
                                q_aligned, r_aligned, offset_q, offset_r = self.smith_waterman(q_gap, r_gap)
                                if q_aligned and r_aligned:  # 确保有有效比对结果
                                    new_q_start = q_end + offset_q
                                    new_q_end = new_q_start + len(q_aligned.replace('-', ''))
                                    new_r_start = r_end + offset_r
                                    new_r_end = new_r_start + len(r_aligned.replace('-', ''))
                                    
                                    # 检查精调后的区间是否合法
                                    if new_q_end <= next_q_start and new_r_end <= next_r_start:
                                        refined.append((new_q_start, new_q_end, new_r_start, new_r_end))
                            except Exception as e:
                                # 如果精调过程出错，跳过此区间
                                print(f"精调区间时出错: {e}")
                
        return refined
    
    def align(self, ref, query):
        """
        执行DNA序列比对的主函数
        
        参数:
        - ref: 参考序列
        - query: 查询序列
        
        返回:
        - 比对结果列表，每个元素是(q_start, q_end, r_start, r_end)的元组
        """
        # 处理输入序列，移除非法字符
        ref = ''.join(c for c in ref.upper() if c in 'ACGT')
        query = ''.join(c for c in query.upper() if c in 'ACGT')
        
        if len(ref) < self.k or len(query) < self.k:
            print(f"警告: 序列长度太短(ref:{len(ref)}, query:{len(query)})，无法进行比对")
            return []
            
        # Step 1: 生成锚点
        anchors = self.find_candidate_anchors(query, ref)
        
        if not anchors:
            print("警告: 未找到候选锚点.")
            # 对于短序列，尝试直接比较
            if len(query) <= 200 and len(ref) <= 200:
                q_align, r_align, q_start, r_start = self.smith_waterman(query, ref)
                q_end = q_start + len(q_align.replace('-', ''))
                r_end = r_start + len(r_align.replace('-', ''))
                if q_end > q_start and r_end > r_start:
                    return [(q_start, q_end, r_start, r_end)]
            return []
        
        # Step 2: 构建图模型
        G = self.build_graph(anchors)
        
        # 检查图是否为有向无环图
        if len(G.nodes()) <= 2:  # 只有S和T节点
            print("警告: 图中只有起点和终点，无法构建有效路径.")
            return []
        
        # Step 3: 动态规划找最长路径
        best_path = self.longest_path_dp(G)
        
        if not best_path or len(best_path) <= 2:  # 只有S和T节点
            print("警告: 未找到有效路径.")
            # 修复：检查锚点有效性后再返回最长锚点
            valid_anchors = []
            for anchor in anchors:
                q_start, q_end, r_start, r_end = anchor
                if q_start >= 0 and q_end <= len(query) and r_start >= 0 and r_end <= len(ref):
                    valid_anchors.append(anchor)
                    
            if valid_anchors:
                longest_anchor = max(valid_anchors, key=lambda x: x[1]-x[0])
                return [longest_anchor]
            return []
        
        # Step 4: 局部精调与输出
        refined = self.refine_alignment(best_path, ref, query)
        
        # 格式化结果
        result = []
        for anchor in refined:
            if anchor not in ("S", "T"):
                q_start, q_end, r_start, r_end = anchor
                # 确保坐标有效
                if (q_start < q_end and r_start < r_end and 
                    q_start >= 0 and q_end <= len(query) and 
                    r_start >= 0 and r_end <= len(ref)):
                    result.append((q_start, q_end, r_start, r_end))
        
        # 确保结果不为空
        if not result and anchors:
            # 失败时退化为返回最好的有效锚点
            valid_anchors = []
            for anchor in anchors:
                q_start, q_end, r_start, r_end = anchor
                if q_start >= 0 and q_end <= len(query) and r_start >= 0 and r_end <= len(ref):
                    valid_anchors.append(anchor)
                    
            if valid_anchors:
                best_anchor = max(valid_anchors, key=lambda x: x[1]-x[0])
                result = [best_anchor]
            
        return sorted(result, key=lambda x: x[0])
