from utils import reverse_complement

class AnchorUtils:
    def __init__(self, aligner):
        """初始化锚点工具类"""
        self.aligner = aligner
    
    def find_anchors(self, query, reference):
        """找到query和reference之间的锚点，使用分层策略"""
        all_anchors = []
        
        # 序列长度调整最大错配数
        if len(query) > 10000 or len(reference) > 10000:
            max_mis = min(self.aligner.max_mismatches, 1)  # 对大序列限制错配
        else:
            max_mis = self.aligner.max_mismatches
            
        # 预处理 - 为长序列创建最小化器索引而不是完整k-mer索引
        use_minimizers = len(reference) > 500000  # 降低最小化器使用阈值
        
        # 分层搜索：避免使用可能不存在的方法
        # 移除对_compute_complexity_profile的直接调用
        if self.aligner.use_tiered_search:
            # 阶段1：使用较长k-mer建立主要锚点
            long_kmers = self.aligner.kmer_sizes[3:]  # 使用较长的k-mer
            main_anchors = self._find_anchors_with_sizes(query, reference, long_kmers, 
                                                        use_minimizers, max_mis)
            
            # 提取已覆盖的query区域
            covered_regions = []
            for anchor in main_anchors:
                covered_regions.append((anchor[0], anchor[1]))
            
            # 阶段2：对未覆盖区域使用较短k-mer
            short_kmers = self.aligner.kmer_sizes[:3]  # 使用较短的k-mer
            gap_regions = self._find_uncovered_regions(covered_regions, len(query))
            
            for start, end in gap_regions:
                if end - start < 20:  # 跳过非常短的区域
                    continue
                    
                # 提取子序列
                sub_query = query[start:end]
                gap_anchors = self._find_anchors_with_sizes(sub_query, reference, short_kmers, 
                                                           use_minimizers, max_mis+1)
                
                # 调整锚点位置并添加到总体锚点列表
                for anchor in gap_anchors:
                    adjusted = (anchor[0]+start, anchor[1]+start, anchor[2], anchor[3], 
                               anchor[4], anchor[5])
                    all_anchors.append(adjusted)
                    
            # 合并两阶段的锚点
            all_anchors.extend(main_anchors)
        else:
            # 传统方法：使用所有k-mer大小
            all_anchors = self._find_anchors_with_sizes(query, reference, self.aligner.kmer_sizes, 
                                                       use_minimizers, max_mis)
        
        # 锚点评分和过滤
        scored_anchors = self._score_and_filter_anchors(all_anchors, query, reference)
        
        return scored_anchors
    
    def _find_uncovered_regions(self, covered_regions, total_length):
        """找出序列中未被锚点覆盖的区域"""
        if not covered_regions:
            return [(0, total_length)]
            
        # 对区域排序
        sorted_regions = sorted(covered_regions)
        
        # 找出间隙
        gaps = []
        last_end = 0
        
        for start, end in sorted_regions:
            if start > last_end:
                gaps.append((last_end, start))
            last_end = max(last_end, end)
            
        # 添加末尾可能的间隙
        if last_end < total_length:
            gaps.append((last_end, total_length))
            
        return gaps
    
    def _find_anchors_with_sizes(self, query, reference, kmer_sizes, use_minimizers, max_mis):
        """使用指定k-mer大小查找锚点"""
        anchors = []
        
        for k_size in kmer_sizes:
            # 优化k-mer索引策略
            ref_index = self.aligner.kmer_utils.build_kmer_index(reference, k_size, use_minimizers)
            
            # 调整步长，较短k-mer使用较小步长
            step = max(1, k_size // 10)
            
            # 查找匹配
            for i in range(0, len(query) - k_size + 1, step):
                kmer = query[i:i+k_size]
                
                # 检查正向匹配（增加模糊匹配灵活度）
                ref_positions = self.aligner.kmer_utils.find_fuzzy_matches(kmer, ref_index, max_mis)
                for ref_pos in ref_positions:
                    # 扩展匹配
                    extended_match = self._extend_match(query, reference, i, ref_pos, k_size, 
                                                      2*max_mis)  # 扩展时允许更多错配
                    if extended_match:
                        anchors.append((*extended_match, "forward"))
                
                # 检查反向互补匹配
                rev_kmer = reverse_complement(kmer)
                rev_positions = self.aligner.kmer_utils.find_fuzzy_matches(rev_kmer, ref_index, max_mis)
                for ref_pos in rev_positions:
                    # 扩展反向互补匹配
                    extended_match = self._extend_reverse_match(query, reference, i, ref_pos, k_size, 
                                                              2*max_mis)
                    if extended_match:
                        anchors.append((*extended_match, "reverse"))
        
        return anchors
    
    def _extend_match(self, query, reference, q_pos, r_pos, k_size, max_errors):
        """改进的匹配延伸策略，使用自适应错误容忍"""
        q_start, r_start = q_pos, r_pos
        q_end, r_end = q_pos + k_size, r_pos + k_size
        
        # 向左扩展
        errors = 0
        while q_start > 0 and r_start > 0:
            if query[q_start-1] != reference[r_start-1]:
                errors += 1
                if errors > max_errors:
                    break
            q_start -= 1
            r_start -= 1
        
        # 向右扩展
        errors = 0
        match_len = q_end - q_start
        while q_end < len(query) and r_end < len(reference):
            if query[q_end] != reference[r_end]:
                errors += 1
                if errors > max_errors:
                    break
            q_end += 1
            r_end += 1
        
        # 添加：延伸中使用动态错误阈值
        # 开始时容忍较少错误，随着匹配长度增加容忍更多错误
        dynamic_max_errors = max(max_errors, int(match_len * 0.08))  # 随长度增加容忍度
        
        # 向右再次尝试延伸，使用动态阈值
        extra_errors = 0
        while q_end < len(query) and r_end < len(reference):
            if query[q_end] != reference[r_end]:
                extra_errors += 1
                if errors + extra_errors > dynamic_max_errors:
                    break
            q_end += 1
            r_end += 1
        
        match_len = q_end - q_start
        if match_len >= self.aligner.min_anchor_length:
            # 计算匹配质量分
            match_score = self._calculate_match_quality(query[q_start:q_end], 
                                                       reference[r_start:r_end])
            return (q_start, q_end-1, r_start, r_end-1, match_score)
        return None
    
    def _extend_reverse_match(self, query, reference, q_pos, r_pos, k_size, max_errors):
        """扩展反向互补匹配"""
        q_start, q_end = q_pos, q_pos + k_size - 1
        r_end, r_start = r_pos + k_size - 1, r_pos
        
        # 向query左侧延伸
        errors = 0
        while q_start > 0 and r_end < len(reference) - 1:
            rc = reverse_complement(query[q_start-1])
            if rc != reference[r_end+1]:
                errors += 1
                if errors > max_errors:
                    break
            q_start -= 1
            r_end += 1
        
        # 向query右侧延伸
        errors = 0
        while q_end < len(query) - 1 and r_start > 0:
            rc = reverse_complement(query[q_end+1])
            if rc != reference[r_start-1]:
                errors += 1
                if errors > max_errors:
                    break
            q_end += 1
            r_start -= 1
        
        match_len = q_end - q_start + 1
        if match_len >= self.aligner.min_anchor_length:
            # 计算匹配质量分
            match_score = self._calculate_reverse_match_quality(query[q_start:q_end+1], 
                                                              reference[r_start:r_end+1])
            return (q_start, q_end, r_start, r_end, match_score)
        return None
    
    def _calculate_match_quality(self, query_seq, ref_seq):
        """计算匹配质量分数"""
        exact_matches = sum(1 for a, b in zip(query_seq, ref_seq) if a == b)
        match_ratio = exact_matches / len(query_seq) if query_seq else 0
        return len(query_seq) * match_ratio
    
    def _calculate_reverse_match_quality(self, query_seq, ref_seq):
        """计算反向互补匹配质量分数"""
        rev_query_rc = reverse_complement(query_seq)
        exact_matches = sum(1 for a, b in zip(rev_query_rc, ref_seq) if a == b)
        match_ratio = exact_matches / len(query_seq) if query_seq else 0
        return len(query_seq) * match_ratio
    
    def _score_and_filter_anchors(self, anchors, query, reference):
        """评分并过滤锚点"""
        # 按分数排序
        scored_anchors = sorted(anchors, key=lambda x: x[4], reverse=True)
        
        # 使用全局一致性提升锚点评分（如果存在）
        try:
            # 检测锚点是否形成一致的匹配模式
            anchor_groups = self._group_consistent_anchors(scored_anchors)
            
            # 提升一致性组内锚点的分数
            boosted_anchors = []
            for anchor in scored_anchors:
                q_start, q_end = anchor[0], anchor[1]
                r_start, r_end = anchor[2], anchor[3]
                score, direction = anchor[4], anchor[5]
                
                # 检查该锚点是否在一个一致性组中
                for group in anchor_groups:
                    if (q_start, q_end) in group:
                        # 提升分数
                        score *= 1.2
                        break
                        
                boosted_anchors.append((q_start, q_end, r_start, r_end, score, direction))
                
            scored_anchors = boosted_anchors
        except:
            # 全局一致性检查失败，跳过此步骤
            pass
        
        # 过滤重叠的锚点
        filtered_anchors = []
        used_query = set()
        
        for anchor in scored_anchors:
            q_start, q_end = anchor[0], anchor[1]
            q_range = set(range(q_start, q_end + 1))
            
            # 检查是否与已使用的query区域重叠
            overlap_ratio = len(q_range.intersection(used_query)) / len(q_range) if q_range else 0
            
            # 允许少量重叠
            if overlap_ratio < 0.5:
                filtered_anchors.append(anchor)
                used_query.update(q_range)
        
        return filtered_anchors
    
    def _group_consistent_anchors(self, anchors, max_distance=1000, min_group_size=3):
        """将锚点分组为一致的匹配模式组
        
        具有相似query-ref距离关系的锚点被认为是一致的，可能代表同一同源区域
        """
        if not anchors or len(anchors) < min_group_size:
            return []
            
        # 计算每个锚点的query-ref距离模式
        anchor_patterns = []
        for i, anchor in enumerate(anchors):
            q_start, r_start = anchor[0], anchor[2]
            q_end, r_end = anchor[1], anchor[3]
            direction = anchor[5]
            
            if direction == "forward":
                # 对于正向匹配，使用相对位置差
                pattern = (q_start - r_start, q_end - r_end)
            else:
                # 对于反向匹配，使用和
                pattern = (q_start + r_start, q_end + r_end)
                
            anchor_patterns.append((i, pattern))
        
        # 根据模式将锚点聚类
        groups = []
        used_anchors = set()
        
        for i, (idx1, pattern1) in enumerate(anchor_patterns):
            if idx1 in used_anchors:
                continue
                
            # 创建新组
            current_group = {(anchors[idx1][0], anchors[idx1][1])}
            used_anchors.add(idx1)
            
            # 查找相似模式的锚点
            for j, (idx2, pattern2) in enumerate(anchor_patterns):
                if idx2 in used_anchors or i == j:
                    continue
                    
                # 检查两个模式是否相似
                if self._patterns_are_similar(pattern1, pattern2, max_distance):
                    current_group.add((anchors[idx2][0], anchors[idx2][1]))
                    used_anchors.add(idx2)
            
            # 只保留足够大的组
            if len(current_group) >= min_group_size:
                groups.append(current_group)
        
        return groups
    
    def _patterns_are_similar(self, pattern1, pattern2, max_distance):
        """检查两个模式是否相似"""
        # 模式是(q-r距离开始, q-r距离结束)元组
        start_diff = abs(pattern1[0] - pattern2[0])
        end_diff = abs(pattern1[1] - pattern2[1])
        
        # 两个差异都应该在允许范围内
        return start_diff <= max_distance and end_diff <= max_distance
