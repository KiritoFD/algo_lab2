import sys
from collections import defaultdict, namedtuple
import bisect

# Define a seed match record
Seed = namedtuple('Seed', ['q_pos', 'r_pos'])


def build_kmer_index(ref: str, k: int) -> dict:
    """
    Build a hash index of k-mers from reference sequence.
    Returns a dict mapping k-mer to list of positions in ref.
    """
    index = defaultdict(list)
    for i in range(len(ref) - k + 1):
        kmer = ref[i:i+k]
        index[kmer].append(i)
    return index


def find_seeds(query: str, ref_index: dict, k: int) -> list:
    """
    Find all exact k-mer matches between query and reference.
    Returns list of Seed(q_pos, r_pos).
    """
    seeds = []
    for i in range(len(query) - k + 1):
        kmer = query[i:i+k]
        for r_pos in ref_index.get(kmer, []):
            seeds.append(Seed(q_pos=i, r_pos=r_pos))
    return seeds


def chain_seeds(seeds: list, max_gap: int = 50) -> list:
    """
    Chain seeds using improved DP with binary search over r_pos to speed up transitions.
    Optimized to O(n log n) using patience sorting-style structure.
    """
    if not seeds:
        return []

    # Sort by query and reference positions
    seeds.sort(key=lambda s: (s.q_pos, s.r_pos))

    # Each chain ends at some seed with increasing q_pos and r_pos
    dp = []  # Stores last r_pos of chains
    chain_ends = []  # Stores the actual seeds in the best chains
    parents = [-1] * len(seeds)
    indices = []  # Stores the index of the seed ending each chain

    for i, s in enumerate(seeds):
        pos = bisect.bisect_left(dp, s.r_pos)
        if pos == len(dp):
            dp.append(s.r_pos)
            indices.append(i)
        elif s.r_pos < dp[pos]:
            dp[pos] = s.r_pos
            indices[pos] = i

        if pos > 0:
            parents[i] = indices[pos - 1]

    # Reconstruct the best chain
    last = indices[-1]
    chain = []
    while last != -1:
        chain.append(seeds[last])
        last = parents[last]
    chain.reverse()
    return chain


def merge_chain(chain: list, k: int) -> list:
    """
    Merge adjacent seeds in chain into longer segments.
    Returns list of (q_start, q_end, r_start, r_end).
    """
    if not chain:
        return []

    segments = []
    start = chain[0]
    q_start, r_start = start.q_pos, start.r_pos
    prev = start

    for s in chain[1:]:
        if s.q_pos == prev.q_pos + 1 and s.r_pos == prev.r_pos + 1:
            prev = s
            continue
        segments.append((q_start, prev.q_pos + k, r_start, prev.r_pos + k))
        q_start, r_start = s.q_pos, s.r_pos
        prev = s

    segments.append((q_start, prev.q_pos + k, r_start, prev.r_pos + k))
    return segments


def align(query: str, ref: str, k: int = 11, max_gap: int = 50) -> list:
    """
    High-level function to compute alignment segments between query and reference.
    """
    ref_index = build_kmer_index(ref, k)
    seeds = find_seeds(query, ref_index, k)
    chain = chain_seeds(seeds, max_gap)
    segments = merge_chain(chain, k)
    return segments


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python dna_align.py <query.fa> <ref.fa>")
        sys.exit(1)

    def read_fasta(path):
        seq = []
        with open(path) as f:
            for line in f:
                if line.startswith('>'):
                    continue
                seq.append(line.strip())
        return ''.join(seq)

    query_seq = read_fasta(sys.argv[1])
    ref_seq = read_fasta(sys.argv[2])
    result = align(query_seq, ref_seq)
    print(result)
