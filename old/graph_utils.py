import numpy as np

class GraphUtils:
    def __init__(self, aligner):
        """初始化图工具类"""
        self.aligner = aligner
    
    def build_graph(self, anchors):
        """改进的图构建，考虑序列结构特征"""
        # 按query起始位置排序锚点
        sorted_anchors = sorted(anchors, key=lambda x: x[0])
        n = len(sorted_anchors)
        
        # 构建邻接表
        graph = [[] for _ in range(n)]
        
        # 对于每对锚点，根据条件评估是否可以连接
        for i in range(n):
            anchor_i = sorted_anchors[i]
            q_end_i, r_end_i, dir_i = anchor_i[1], anchor_i[3], anchor_i[5]
            
            for j in range(i + 1, n):
                anchor_j = sorted_anchors[j]
                q_start_j, r_start_j, dir_j = anchor_j[0], anchor_j[2], anchor_j[5]
                
                # 检查query上的距离
                q_gap = q_start_j - q_end_i - 1
                
                # 只链接在合理距离内的锚点
                if 0 <= q_gap <= self.aligner.max_gap:
                    # 使用锚点分数和关系质量作为边的权重
                    weight = anchor_j[4]
                    
                    # 如果方向一致，增加权重
                    if dir_i == dir_j:
                        weight *= 1.4
                    
                    # 改为使用简单的边权重计算，避免调用不存在的方法
                    # 基于query和reference间隙比例计算一致性得分
                    if dir_i == "forward" and dir_j == "forward":
                        # 计算reference上的间隙
                        r_gap = r_start_j - r_end_i - 1
                        
                        # 计算间隙比例相似度
                        if q_gap > 0 and r_gap > 0:
                            gap_ratio_diff = abs(q_gap/r_gap - 1.0)
                            consistency = max(0, 1.0 - gap_ratio_diff)
                            weight *= (1.0 + 0.3 * consistency)
                    
                    # 添加边
                    graph[i].append((j, weight))
        
        return graph, sorted_anchors
    
    def _calculate_edge_smoothness(self, anchor1, anchor2):
        """计算两个锚点之间边的平滑度"""
        # 提取锚点信息
        q_end_1, r_end_1 = anchor1[1], anchor1[3]
        q_start_2, r_start_2 = anchor2[0], anchor2[2]
        dir_1, dir_2 = anchor1[5], anchor2[5]
        
        # 默认平滑度中等
        smoothness = 0.5
        
        # 如果方向一致，根据间隙比例计算平滑度
        if dir_1 == dir_2:
            q_gap = q_start_2 - q_end_1 - 1
            
            if dir_1 == "forward":
                r_gap = r_start_2 - r_end_1 - 1
                # 计算间隙比率差异
                if q_gap > 0 and r_gap > 0:
                    ratio_diff = abs(q_gap/r_gap - 1.0)
                    # 比率接近1表示高平滑度
                    smoothness = max(0, 1.0 - min(ratio_diff, 1.0))
        
        return smoothness
    
    def _analyze_mapping_trend(self, anchors):
        """简化版映射趋势分析，返回平均每个query位置对应多少ref位置"""
        if not anchors:
            return 1.0
            
        # 计算正向匹配的平均比例
        forward_ratios = []
        for anchor in anchors:
            if anchor[5] == "forward":
                q_len = anchor[1] - anchor[0] + 1
                r_len = anchor[3] - anchor[2] + 1
                if q_len > 0:
                    forward_ratios.append(r_len / q_len)
        
        # 如果没有足够的正向匹配，返回默认值
        if not forward_ratios or len(forward_ratios) < 3:
            return 1.0
            
        # 删除极端值
        forward_ratios.sort()
        trimmed_ratios = forward_ratios[1:-1] if len(forward_ratios) > 4 else forward_ratios
        
        # 返回平均比例
        return sum(trimmed_ratios) / len(trimmed_ratios)
    
    def _calculate_trend_consistency(self, pos1, pos2, trend):
        """计算两个位置与总体趋势的一致性"""
        q_end, r_end = pos1
        q_start, r_start = pos2
        
        # 计算query和reference的变化
        q_change = q_start - q_end
        r_change = r_start - r_end
        
        if q_change <= 0:
            return 0
            
        # 计算实际比例与趋势比例的一致性
        actual_ratio = r_change / q_change if q_change != 0 else 0
        ratio_diff = abs(actual_ratio / trend - 1.0) if trend != 0 else 1.0
        
        # 转换为0到1之间的一致性分数
        consistency = max(0, 1.0 - min(ratio_diff, 1.0))
        return consistency
    
    def find_longest_path(self, graph, anchors):
        """增强版最长路径查找，考虑全局特征"""
        n = len(graph)
        if n == 0:
            return []
        
        # dp[i] 表示从起点到i的最长路径长度
        dp = [anchors[i][4] for i in range(n)]  # 初始化为每个锚点的分数
        prev = [-1] * n  # 用于回溯路径
        
        # 添加：路径平滑度评分
        # 计算每个锚点的"平滑度分数"，即与相邻锚点的一致性
        smoothness_scores = [0] * n
        
        for i in range(n):
            # 对于每个节点，检查其所有出边
            total_smoothness = 0
            edge_count = 0
            
            for j, _ in graph[i]:
                # 计算这条边的平滑度
                edge_smoothness = self._calculate_edge_smoothness(anchors[i], anchors[j])
                total_smoothness += edge_smoothness
                edge_count += 1
                
            if edge_count > 0:
                smoothness_scores[i] = total_smoothness / edge_count
        
        # 在动态规划中融入平滑度分数
        for i in range(n):
            for j, weight in graph[i]:
                # 调整权重，考虑目标节点的平滑度
                adjusted_weight = weight * (1 + 0.2 * smoothness_scores[j])
                
                if dp[i] + adjusted_weight > dp[j]:
                    dp[j] = dp[i] + adjusted_weight
                    prev[j] = i
        
        # 找到终点（最长路径的最后一个节点）
        end_node = np.argmax(dp)
        
        # 回溯构建路径
        path = []
        while end_node != -1:
            path.append(end_node)
            end_node = prev[end_node]
        
        path.reverse()
        return path
    
    def find_multiple_paths(self, graph, anchors, max_paths=3):
        """寻找多条非重叠的最优路径"""
        n = len(graph)
        if n == 0:
            return []
        
        # 第一条路径 - 常规动态规划
        dp = [anchors[i][4] for i in range(n)]
        prev = [-1] * n
        
        for i in range(n):
            for j, weight in graph[i]:
                if dp[i] + weight > dp[j]:
                    dp[j] = dp[i] + weight
                    prev[j] = i
        
        # 构建路径
        paths = []
        used_nodes = set()
        
        # 找出得分最高的终点
        end_nodes = sorted(range(n), key=lambda i: dp[i], reverse=True)
        
        for end_node in end_nodes[:3]:  # 尝试前3个最高得分终点
            if dp[end_node] < self.aligner.min_chain_score or end_node in used_nodes:
                continue
                
            path = []
            curr = end_node
            while curr != -1 and curr not in used_nodes:
                path.append(curr)
                used_nodes.add(curr)
                curr = prev[curr]
                
            if path:
                path.reverse()
                paths.append(path)
                
                # 最多找max_paths条路径
                if len(paths) >= max_paths:
                    break
        
        return paths
