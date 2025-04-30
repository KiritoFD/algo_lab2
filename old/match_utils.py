from utils import reverse_complement

class MatchUtils:
    def __init__(self, aligner):
        """初始化匹配工具类"""
        self.aligner = aligner
    
    def merge_adjacent_matches(self, matches):
        """合并相邻且可以合并的匹配"""
        if not matches:
            return []
        
        # 按query起始位置排序
        sorted_matches = sorted(matches, key=lambda x: x[0])
        merged = [sorted_matches[0]]
        
        for current in sorted_matches[1:]:
            prev = merged[-1]
            
            # 检查是否可合并
            q_gap = current[0] - prev[1] - 1
            
            # 检查reference方向
            prev_forward = prev[2] <= prev[3]
            current_forward = current[2] <= current[3]
            
            # 如果方向相同且间隔合理
            if prev_forward == current_forward and 0 <= q_gap <= 20:  # 增大间隔阈值
                if prev_forward:  # 都是正向匹配
                    r_gap = current[2] - prev[3] - 1
                    # 合并条件：间隙类似或很小
                    if -10 <= r_gap - q_gap <= 10 or (r_gap <= 10 and q_gap <= 10):  # 更宽松的条件
                        merged[-1] = (prev[0], current[1], prev[2], current[3])
                        continue
                else:  # 都是反向匹配
                    r_gap = prev[2] - current[3] - 1
                    if -10 <= r_gap - q_gap <= 10 or (r_gap <= 10 and q_gap <= 10):
                        merged[-1] = (prev[0], current[1], prev[2], current[3])
                        continue
            
            # 不能合并，添加为新的匹配
            merged.append(current)
                
        return merged
    
    def detect_special_variants(self, matches, query, reference):
        """增强的特殊变异检测"""
        if not matches:
            return []
        
        result = []
        last_q_end = -1
        
        for i, match in enumerate(matches):
            q_start, q_end, r_start, r_end = match
            
            # 跳过重叠匹配
            if q_start <= last_q_end:
                continue
            
            # 添加当前匹配
            result.append(match)
            last_q_end = q_end
            
            # 检查是否有未匹配的区域
            if i < len(matches) - 1:
                next_match = matches[i+1]
                next_q_start = next_match[0]
                
                # 如果query中有较大的未匹配区域
                if next_q_start - q_end > 20:
                    unmatched_query = query[q_end+1:next_q_start]
                    
                    # 尝试在reference中查找此未匹配区域
                    pos = reference.find(unmatched_query)
                    if pos >= 0:
                        # 找到了匹配，添加为新的匹配区域
                        result.append((q_end+1, next_q_start-1, pos, pos+len(unmatched_query)-1))
        
        # 添加：使用局部比对识别小型结构变异
        for i in range(len(result)-1):
            curr = result[i]
            next_match = result[i+1]
            
            q_gap_start = curr[1] + 1
            q_gap_end = next_match[0] - 1
            q_gap_size = q_gap_end - q_gap_start + 1
            
            # 只处理中等大小的间隙
            if 10 <= q_gap_size <= 100:
                # 提取间隙序列
                gap_seq = query[q_gap_start:q_gap_end+1]
                
                # 检查是否为潜在的结构变异
                variant_type, positions = self._detect_structural_variant(
                    gap_seq, curr, next_match, reference
                )
                
                if variant_type and positions:
                    # 根据变异类型添加匹配
                    if variant_type == "inversion":
                        q_start, q_end, r_start, r_end = positions
                        # 对于倒位，r_start > r_end
                        special_match = (q_start, q_end, r_end, r_start)
                        result.insert(i+1, special_match)
                    elif variant_type == "translocation":
                        q_start, q_end, r_start, r_end = positions
                        special_match = (q_start, q_end, r_start, r_end)
                        result.insert(i+1, special_match)
        
        return result
    
    def fill_gaps(self, matches, query, reference, max_size=500):
        """改进的间隙填补策略"""
        if not matches:
            return []
            
        result = []
        sorted_matches = sorted(matches, key=lambda x: x[0])
        
        # 添加第一个匹配
        result.append(sorted_matches[0])
        
        for i in range(1, len(sorted_matches)):
            prev = result[-1]
            curr = sorted_matches[i]
            
            # 计算gap大小
            q_gap_start = prev[1] + 1
            q_gap_end = curr[0] - 1
            q_gap_size = q_gap_end - q_gap_start + 1
            
            # 对适当大小的gap尝试填充
            if 20 <= q_gap_size <= max_size:
                gap_seq = query[q_gap_start:q_gap_end+1]
                
                # 尝试查找匹配
                self._try_fill_gap(gap_seq, q_gap_start, reference, result)
            
            # 添加当前匹配
            result.append(curr)
        
        # 按query位置排序
        return sorted(result, key=lambda x: x[0])
    
    def _try_fill_gap(self, gap_seq, q_gap_start, reference, result):
        """尝试为间隙序列寻找匹配"""
        # 1. 尝试直接查找完全匹配
        if len(gap_seq) < 50:
            for w_size in range(len(gap_seq), max(15, len(gap_seq) - 10), -1):
                for w_start in range(len(gap_seq) - w_size + 1):
                    window = gap_seq[w_start:w_start+w_size]
                    pos = reference.find(window)
                    if pos >= 0:
                        q_match_start = q_gap_start + w_start
                        q_match_end = q_match_start + w_size - 1
                        r_match_start = pos
                        r_match_end = pos + w_size - 1
                        result.append((q_match_start, q_match_end, r_match_start, r_match_end))
                        return True
        
        # 2. 尝试反向互补匹配
        if len(gap_seq) < 50:
            gap_seq_rc = reverse_complement(gap_seq)
            for w_size in range(len(gap_seq), max(15, len(gap_seq) - 10), -1):
                for w_start in range(len(gap_seq) - w_size + 1):
                    window = gap_seq_rc[w_start:w_start+w_size]
                    pos = reference.find(window)
                    if pos >= 0:
                        q_match_start = q_gap_start + w_start
                        q_match_end = q_match_start + w_size - 1
                        r_match_end = pos + w_size - 1
                        r_match_start = pos
                        result.append((q_match_start, q_match_end, r_match_end, r_match_start))
                        return True
        
        return False
    
    def _detect_structural_variant(self, gap_seq, curr, next_match, reference):
        """根据间隙序列和相邻匹配检测潜在的结构变异"""
        # 这里实现具体的结构变异检测逻辑
        # 返回值示例：("inversion", (q_start, q_end, r_start, r_end))
        # 实际实现中需要根据具体需求进行变更
        
        return None, None
    
    def consolidate_matches(self, matches):
        """合并太近的匹配以改善整体覆盖率"""
        if len(matches) <= 1:
            return matches
            
        # 按query位置排序
        sorted_matches = sorted(matches, key=lambda x: x[0])
        result = [sorted_matches[0]]
        
        for curr in sorted_matches[1:]:
            prev = result[-1]
            
            # 计算间隔
            q_gap = curr[0] - prev[1] - 1
            
            # 如果间隔非常小，尝试合并
            if q_gap <= 5:
                # 检查方向一致性
                prev_forward = prev[2] <= prev[3]
                curr_forward = curr[2] <= curr[3]
                
                if prev_forward == curr_forward:
                    # 合并匹配
                    result[-1] = (prev[0], curr[1], prev[2], curr[3])
                    continue
            
            # 不能合并，作为新的匹配添加
            result.append(curr)
        
        return result
