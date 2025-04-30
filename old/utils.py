def reverse_complement(seq):
    """计算DNA序列的反向互补序列"""
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 
                 'a': 't', 'c': 'g', 'g': 'c', 't': 'a'}
    return ''.join(complement.get(base, base) for base in reversed(seq))

def read_fasta(file_path):
    """读取FASTA格式文件"""
    sequences = {}
    current_seq_name = None
    current_seq = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_seq_name:
                    sequences[current_seq_name] = ''.join(current_seq)
                current_seq_name = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
    
    if current_seq_name:
        sequences[current_seq_name] = ''.join(current_seq)
    
    return sequences
