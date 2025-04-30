from collections import defaultdict, Counter
from utils import reverse_complement
import numpy as np

class KmerUtils:
    def __init__(self, aligner):
        """初始化k-mer工具类"""
        self.aligner = aligner
        # 添加自适应k-mer大小策略
        self.adaptive_kmer_sizes = True
        # 动态调整最小化器窗口大小
        self.dynamic_window_sizing = True
        # 添加缓存避免重复计算
        self._kmer_index_cache = {}
        # 对不同区域使用不同策略
        self.region_specific_strategies = True
        # 高频k-mer阈值，超过此频率的k-mer被认为是重复区域
        self.high_freq_threshold = 100
        
    def compute_minimizers(self, seq, w, k):
        """优化的最小化器计算，使用自适应窗口大小"""
        # 如果启用动态窗口调整，根据序列复杂度调整窗口
        if self.dynamic_window_sizing:
            complexity = self._estimate_local_complexity(seq[:min(1000, len(seq))])
            # 复杂区域使用较小窗口增加敏感度
            if complexity > 0.7:
                w = max(3, w-2)
            # 低复杂度区域使用较大窗口减少冗余
            elif complexity < 0.3:
                w = min(w+2, k)
        
        minimizers = {}
        if len(seq) < k:
            return minimizers
        
        # 优化：使用滚动哈希算法加速
        # 计算初始窗口内的哈希值
        window_hashes = []
        for j in range(min(w, len(seq) - k + 1)):
            if j + k <= len(seq):
                kmer = seq[j:j+k]
                # 使用自定义DNA哈希函数，比通用hash函数更适合DNA序列
                h = self._dna_hash(kmer)
                window_hashes.append((h, kmer, j))
        
        # 处理所有可能的窗口
        for i in range(len(seq) - w - k + 2):
            # 找到当前窗口中哈希值最小的k-mer
            min_hash_data = min(window_hashes, key=lambda x: x[0])
            min_hash, min_kmer, min_pos = min_hash_data
            
            # 记录最小化器
            actual_pos = i + min_pos
            if min_kmer not in minimizers:
                minimizers[min_kmer] = []
            minimizers[min_kmer].append(actual_pos)
            
            # 滑动窗口：移除第一个k-mer，添加新的k-mer
            if i + w < len(seq) - k + 1:
                # 移除第一个k-mer
                window_hashes = [(h, km, p-1) for h, km, p in window_hashes if p > 0]
                
                # 添加新的k-mer
                new_kmer = seq[i+w:i+w+k]
                new_hash = self._dna_hash(new_kmer)
                window_hashes.append((new_hash, new_kmer, w-1))
        
        # 过滤高频k-mer
        return self._filter_high_frequency_kmers(minimizers)
    
    def _dna_hash(self, kmer):
        """专门为DNA设计的哈希函数，提供更好的分布"""
        # 使用Rabin-Karp滚动哈希算法的变种
        base = 4  # DNA字母表大小
        h = 0
        for ch in kmer:
            code = {'A': 0, 'C': 1, 'G': 2, 'T': 3}.get(ch, 0)
            h = h * base + code
        return h
    
    def _filter_high_frequency_kmers(self, kmer_dict):
        """过滤掉高频k-mer（重复区域）"""
        filtered = {}
        for kmer, positions in kmer_dict.items():
            if len(positions) < self.high_freq_threshold:
                filtered[kmer] = positions
        return filtered
    
    def find_fuzzy_matches(self, kmer, reference_kmers, max_mismatches=1):
        """改进的模糊匹配策略"""
        if max_mismatches == 0 or len(kmer) <= max_mismatches * 3:
            # 如果不允许错配或kmer太短，只返回精确匹配
            return reference_kmers.get(kmer, [])
        
        # 对于短k-mer，使用更严格的错配标准
        if len(kmer) < 12:
            max_mismatches = min(max_mismatches, 1)
        
        matches = []
        # 先检查精确匹配
        if kmer in reference_kmers:
            matches.extend(reference_kmers[kmer])
        
        # 使用种子扩展方法，先找到部分精确匹配（种子）
        if max_mismatches > 0 and len(kmer) >= 8:
            seed_len = len(kmer) // (max_mismatches + 1)
            for i in range(max_mismatches + 1):
                seed_start = i * seed_len
                if seed_start + seed_len > len(kmer):
                    break
                seed = kmer[seed_start:seed_start+seed_len]
                
                # 在所有k-mer中寻找包含这个种子的k-mer
                for ref_kmer in reference_kmers:
                    if len(ref_kmer) != len(kmer):
                        continue
                    
                    if ref_kmer[seed_start:seed_start+seed_len] == seed:
                        # 检查完整k-mer的错配数
                        mismatches = sum(1 for a, b in zip(kmer, ref_kmer) if a != b)
                        if mismatches <= max_mismatches:
                            matches.extend(reference_kmers[ref_kmer])
        
        # 添加：为长k-mer使用分段匹配策略，提高容错率
        if len(kmer) >= 15 and max_mismatches > 0:
            # 将k-mer分为3段，只要有2段匹配良好即可认为有潜在匹配
            segment_len = len(kmer) // 3
            for ref_kmer in reference_kmers:
                if len(ref_kmer) != len(kmer):
                    continue
                
                # 计算每段的匹配情况
                segment_matches = [
                    sum(1 for a, b in zip(kmer[i:i+segment_len], ref_kmer[i:i+segment_len]) if a == b)
                    for i in range(0, len(kmer), segment_len)
                ]
                
                # 如果有至少两段匹配度高，考虑为潜在匹配
                good_segments = sum(1 for m in segment_matches if m >= segment_len * 0.8)
                if good_segments >= 2:
                    # 检查整体错配数
                    total_mismatches = sum(1 for a, b in zip(kmer, ref_kmer) if a != b)
                    if total_mismatches <= max_mismatches * 1.5:  # 稍微放宽条件
                        matches.extend(reference_kmers[ref_kmer])
        
        return matches
    
    def build_kmer_index(self, sequence, k_size, use_minimizers=False):
        """为序列构建k-mer索引，优化索引策略"""
        # 调试代码
        print(f"build_kmer_index called with k_size={k_size}, use_minimizers={use_minimizers}")
        print(f"self.high_freq_threshold exists: {'high_freq_threshold' in dir(self)}")
        print(f"self.__dict__: {self.__dict__}")
        
        index = defaultdict(list)
        
        # 根据序列长度动态调整策略
        if len(sequence) > 1000000:
            # 对非常长的序列，使用采样策略
            step = max(1, k_size // 3)
        else:
            step = 1
            
        if use_minimizers:
            # 使用最小化器减少索引大小
            minimizers = self.compute_minimizers(sequence, self.aligner.minimizer_window, k_size)
            for kmer, positions in minimizers.items():
                for pos in positions:
                    index[kmer].append(pos)
        else:
            # 使用优化的k-mer采样策略
            # 统计所有k-mer频率
            all_kmers = Counter()
            for i in range(0, len(sequence) - k_size + 1, step):
                kmer = sequence[i:i+k_size]
                all_kmers[kmer] += 1
            
            # 先将频率适中的k-mer添加到索引
            for i in range(0, len(sequence) - k_size + 1, step):
                kmer = sequence[i:i+k_size]
                if all_kmers[kmer] < self.high_freq_threshold:  # 过滤高频k-mer
                    index[kmer].append(i)
                    
            # 稀有k-mer优先建立索引，对于中长度序列也可以提高匹配质量
            if len(sequence) < 500000:
                # 找出频率较低的"稀有"k-mer
                rare_kmers = {k for k, v in all_kmers.items() if 1 <= v <= 3}
                for i in range(0, len(sequence) - k_size + 1):
                    kmer = sequence[i:i+k_size]
                    if kmer in rare_kmers and i % step != 0:  # 不重复添加
                        index[kmer].append(i)
        
        return index
    
    def compute_adaptive_kmers(self, sequence, kmer_sizes):
        """计算序列的自适应k-mer索引，根据序列复杂度选择不同k值"""
        result = {}
        # 计算序列复杂度
        complexity = self._estimate_sequence_complexity(sequence)
        
        # 根据复杂度选择合适的k-mer大小
        selected_k = kmer_sizes[0]  # 默认使用最小的k
        if complexity > 0.7:  # 高复杂度，使用较短的k
            selected_k = kmer_sizes[0]
        elif complexity > 0.5:  # 中等复杂度
            selected_k = kmer_sizes[1]
        else:  # 低复杂度，使用较长的k
            selected_k = kmer_sizes[2]
            
        # 构建索引
        result[selected_k] = self.build_kmer_index(sequence, selected_k, False)
        return result, selected_k
    
    def _estimate_sequence_complexity(self, sequence, sample_size=1000):
        """估计序列的复杂度"""
        if len(sequence) <= sample_size:
            sample = sequence
        else:
            # 取多个区域的样本
            samples = []
            step = len(sequence) // 5
            for i in range(0, len(sequence) - sample_size//5, step):
                samples.append(sequence[i:i+sample_size//5])
            sample = ''.join(samples)
        
        # 计算k=3的k-mer多样性
        kmers = set()
        for i in range(len(sample) - 3 + 1):
            kmers.add(sample[i:i+3])
            
        # 计算熵作为复杂度衡量
        max_possible = min(4**3, len(sample) - 3 + 1)
        complexity = len(kmers) / max_possible
        
        return complexity
    
    def _estimate_local_complexity(self, sequence, kmer_size=3):
        """估计局部序列复杂度
        
        使用短k-mer的分布来评估局部序列的复杂度
        """
        if len(sequence) < kmer_size:
            return 0.5  # 默认中等复杂度
            
        # 计算k-mer的频率分布
        kmer_counts = defaultdict(int)
        for i in range(len(sequence) - kmer_size + 1):
            kmer = sequence[i:i+kmer_size]
            kmer_counts[kmer] += 1
            
        # 计算归一化熵作为复杂度指标
        total_kmers = len(sequence) - kmer_size + 1
        entropy = 0
        for count in kmer_counts.values():
            p = count / total_kmers
            entropy -= p * (np.log(p) if p > 0 else 0)
            
        # 归一化熵，使结果在0-1之间
        max_entropy = min(np.log(4**kmer_size), np.log(total_kmers))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.5
        
        return normalized_entropy
