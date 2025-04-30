import numpy as np
from collections import defaultdict
from utils import reverse_complement
from kmer_utils import KmerUtils
from anchor_utils import AnchorUtils
from graph_utils import GraphUtils
from match_utils import MatchUtils

class DNAAligner:
    def __init__(self, k=8, min_anchor_length=12, min_chain_score=20, max_gap=2000):
        """初始化DNA序列比对器"""
        # 根据输入序列长度动态调整参数
        self.k = k
        self.min_anchor_length = min_anchor_length
        self.min_chain_score = min_chain_score
        self.max_gap = max_gap
        # 使用更广范围的k-mer大小
        self.kmer_sizes = [k, k+2, k+4, k+7, k+10, k+15]
        # 允许更多错配以提高敏感度
        self.max_mismatches = 2
        # 最小化器窗口基准大小
        self.minimizer_window = 7
        # 启用分层搜索和区域特异性策略
        self.use_tiered_search = True
        
        # 确保kmer_utils在其他工具类之前初始化
        self.kmer_utils = KmerUtils(self)
        self.anchor_utils = AnchorUtils(self)
        self.graph_utils = GraphUtils(self)
        self.match_utils = MatchUtils(self)
        
        # 添加明确的high_freq_threshold以确保可访问
        self.high_freq_threshold = 100
    
    def find_anchors(self, query, reference):
        """找到query和reference之间的锚点"""
        return self.anchor_utils.find_anchors(query, reference)
    
    def build_graph(self, anchors):
        """构建锚点图"""
        return self.graph_utils.build_graph(anchors)
    
    def find_longest_path(self, graph, anchors):
        """使用动态规划找到最长路径"""
        return self.graph_utils.find_longest_path(graph, anchors)
    
    def align(self, query, reference):
        """增强版序列比对流程"""
        # 找到锚点
        anchors = self.find_anchors(query, reference)
        if not anchors:
            return []
        
        # 构建图
        graph, sorted_anchors = self.build_graph(anchors)
        
        # 寻找最长路径
        path = self.find_longest_path(graph, sorted_anchors)
        
        # 转换为输出格式
        result = []
        for node in path:
            anchor = sorted_anchors[node]
            q_start, q_end, r_start, r_end = anchor[0], anchor[1], anchor[2], anchor[3]
            
            # 确保正确的顺序
            if anchor[5] == "forward":
                result.append((q_start, q_end, r_start, r_end))
            else:  # reverse
                result.append((q_start, q_end, r_start, r_end))
        
        # 合并相邻匹配
        merged = self.match_utils.merge_adjacent_matches(result)
        
        # 检测特殊变异类型
        final_result = self.match_utils.detect_special_variants(merged, query, reference)
        
        # 尝试填补间隙
        enhanced_matches = self.match_utils.fill_gaps(final_result, query, reference)
        
        # 合并过近的匹配
        return self.match_utils.consolidate_matches(enhanced_matches)
    
    # 添加一个方法直接检查属性是否存在
    def check_initialization(self):
        """检查各个组件是否正确初始化"""
        try:
            print(f"KmerUtils high_freq_threshold: {self.kmer_utils.high_freq_threshold}")
            print(f"k-mer sizes: {self.kmer_sizes}")
            return True
        except Exception as e:
            print(f"初始化检查失败: {e}")
            return False
