#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Define EVAL_INCLUDED to prevent multiple main functions
#define EVAL_INCLUDED 1

// Forward declarations for functions from run.c and eval.c
extern void chain_anchors_kernel(int n_anchors, 
                          int* q_s_arr, int* q_e_arr, 
                          int* r_s_arr, int* r_e_arr, 
                          char* strand_arr, 
                          int kmersize, 
                          int max_gap_between_anchors, 
                          int max_diagonal_difference, 
                          int max_allowed_overlap,
                          int* dp_score,
                          int* parent_idx);

extern int select_segments_kernel(int n_segs, 
                           int* seg_q_s_arr, 
                           int* seg_q_e_arr, 
                           int* seg_scores_arr,
                           int* dp_select_score,
                           int* prev_select_idx,
                           int* selected_indices);

extern int calculate_value(const char* tuples_str, const char* ref, const char* query);
extern void format_tuples_for_display(const char* tuples_str);
extern unsigned long hash_kmer(const char* kmer);

// Define a map structure for nucleotide reverse complementing
typedef struct {
    char from;
    char to;
} NucleotideMap;

extern NucleotideMap rc_map[];
extern void get_rc(const char* seq, char* result);

/**
 * Read a sequence from a file
 * 
 * @param filename The file to read from
 * @return The sequence as a string (must be freed by caller)
 */
char* read_sequence_from_file(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file %s\n", filename);
        return NULL;
    }
    
    // Get file size
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    // Allocate memory for the sequence (add 1 for null terminator)
    char* sequence = (char*)malloc(file_size + 1);
    if (!sequence) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(file);
        return NULL;
    }
    
    // Read the sequence
    size_t bytes_read = fread(sequence, 1, file_size, file);
    sequence[bytes_read] = '\0';
    
    // Remove any newlines or spaces
    char* cleaned = (char*)malloc(bytes_read + 1);
    if (!cleaned) {
        fprintf(stderr, "Memory allocation failed\n");
        free(sequence);
        fclose(file);
        return NULL;
    }
    
    int j = 0;
    for (int i = 0; i < bytes_read; i++) {
        if (sequence[i] != '\n' && sequence[i] != ' ' && sequence[i] != '\r') {
            cleaned[j++] = sequence[i];
        }
    }
    cleaned[j] = '\0';
    
    free(sequence);
    fclose(file);
    return cleaned;
}

/**
 * Simple implementation of Python's seq2hashtable_multi_test function
 */
typedef struct {
    int q_s;
    int r_s;
    int strand;
    int kmersize;
} Anchor;

Anchor* seq2hashtable_multi_test(const char* refseq, const char* testseq, int kmersize, int shift, int* num_anchors) {
    char* rc_testseq = (char*)malloc(strlen(testseq) + 1);
    get_rc(testseq, rc_testseq);
    
    int testseq_len = strlen(testseq);
    int refseq_len = strlen(refseq);
    
    // Create a simple hash table for kmers in the reference
    typedef struct {
        unsigned long hash;
        int position;
        struct HashEntry* next;
    } HashEntry;
    
    HashEntry** hash_table = (HashEntry**)calloc(10000, sizeof(HashEntry*));
    
    // Calculate hash for 'N' * kmersize to skip
    char* n_kmer = (char*)malloc(kmersize + 1);
    memset(n_kmer, 'N', kmersize);
    n_kmer[kmersize] = '\0';
    unsigned long skip_hash = hash_kmer(n_kmer);
    free(n_kmer);
    
    // Build hash table for reference kmers
    for (int iloc = 0; iloc <= refseq_len - kmersize; iloc++) {
        char* kmer = (char*)malloc(kmersize + 1);
        strncpy(kmer, refseq + iloc, kmersize);
        kmer[kmersize] = '\0';
        
        unsigned long hash_val = hash_kmer(kmer);
        free(kmer);
        
        if (hash_val == skip_hash) continue;
        
        // Add to hash table
        int bucket = hash_val % 10000;
        HashEntry* entry = (HashEntry*)malloc(sizeof(HashEntry));
        entry->hash = hash_val;
        entry->position = iloc;
        entry->next = hash_table[bucket];
        hash_table[bucket] = entry;
    }
    
    // Allocate anchors array (we'll realloc as needed)
    Anchor* anchors = (Anchor*)malloc(1000 * sizeof(Anchor));
    int capacity = 1000;
    *num_anchors = 0;
    
    // Scan the test sequence for matches
    for (int iloc = 0; iloc <= testseq_len - kmersize; iloc += shift) {
        // Forward strand
        char* kmer = (char*)malloc(kmersize + 1);
        strncpy(kmer, testseq + iloc, kmersize);
        kmer[kmersize] = '\0';
        
        unsigned long hash_val = hash_kmer(kmer);
        free(kmer);
        
        // Check for matches in reference
        int bucket = hash_val % 10000;
        HashEntry* entry = hash_table[bucket];
        while (entry) {
            if (entry->hash == hash_val) {
                // Found a match
                if (*num_anchors >= capacity) {
                    capacity *= 2;
                    anchors = (Anchor*)realloc(anchors, capacity * sizeof(Anchor));
                }
                
                anchors[*num_anchors].q_s = iloc;
                anchors[*num_anchors].r_s = entry->position;
                anchors[*num_anchors].strand = 1;
                anchors[*num_anchors].kmersize = kmersize;
                (*num_anchors)++;
            }
            entry = entry->next;
        }
        
        // Reverse complement strand
        char* rc_kmer = (char*)malloc(kmersize + 1);
        strncpy(rc_kmer, rc_testseq + (testseq_len - (iloc + kmersize)), kmersize);
        rc_kmer[kmersize] = '\0';
        
        hash_val = hash_kmer(rc_kmer);
        free(rc_kmer);
        
        // Check for matches in reference
        bucket = hash_val % 10000;
        entry = hash_table[bucket];
        while (entry) {
            if (entry->hash == hash_val) {
                // Found a match
                if (*num_anchors >= capacity) {
                    capacity *= 2;
                    anchors = (Anchor*)realloc(anchors, capacity * sizeof(Anchor));
                }
                
                anchors[*num_anchors].q_s = iloc;
                anchors[*num_anchors].r_s = entry->position;
                anchors[*num_anchors].strand = -1;
                anchors[*num_anchors].kmersize = kmersize;
                (*num_anchors)++;
            }
            entry = entry->next;
        }
    }
    
    // Clean up hash table
    for (int i = 0; i < 10000; i++) {
        HashEntry* entry = hash_table[i];
        while (entry) {
            HashEntry* next = entry->next;
            free(entry);
            entry = next;
        }
    }
    free(hash_table);
    free(rc_testseq);
    
    return anchors;
}

/**
 * Implementation of the Python function to process alignment
 */
char* process_alignment(Anchor* anchors, int num_anchors, int max_gap_param, int max_diag_diff_param, 
                     float overlap_factor_param, int min_anchors_param) {
    if (num_anchors == 0) {
        return strdup("");
    }
    
    int kmersize = anchors[0].kmersize;
    
    // Create arrays for anchor chaining
    int* q_s_arr = (int*)malloc(num_anchors * sizeof(int));
    int* q_e_arr = (int*)malloc(num_anchors * sizeof(int));
    int* r_s_arr = (int*)malloc(num_anchors * sizeof(int));
    int* r_e_arr = (int*)malloc(num_anchors * sizeof(int));
    char* strand_arr = (char*)malloc(num_anchors * sizeof(char));
    
    for (int i = 0; i < num_anchors; i++) {
        q_s_arr[i] = anchors[i].q_s;
        q_e_arr[i] = anchors[i].q_s + kmersize;
        r_s_arr[i] = anchors[i].r_s;
        r_e_arr[i] = anchors[i].r_s + kmersize;
        strand_arr[i] = anchors[i].strand;
    }
    
    // Chaining anchors
    int* dp_score = (int*)malloc(num_anchors * sizeof(int));
    int* parent_idx = (int*)malloc(num_anchors * sizeof(int));
    
    int max_allowed_overlap = (int)(kmersize * overlap_factor_param);
    
    chain_anchors_kernel(num_anchors, q_s_arr, q_e_arr, r_s_arr, r_e_arr, strand_arr,
                         kmersize, max_gap_param, max_diag_diff_param, max_allowed_overlap,
                         dp_score, parent_idx);
    
    // Form candidate segments
    typedef struct {
        int q_s, q_e, r_s, r_e;
        int score;
        char strand;
    } Segment;
    
    Segment* segments = NULL;
    int segment_count = 0;
    
    for (int i = 0; i < num_anchors; i++) {
        // Trace the chain
        int chain_length = 0;
        int curr = i;
        while (curr != -1) {
            chain_length++;
            curr = parent_idx[curr];
        }
        
        if (chain_length >= min_anchors_param) {
            // Collect chain indices
            int* chain_indices = (int*)malloc(chain_length * sizeof(int));
            curr = i;
            int idx = 0;
            while (curr != -1) {
                chain_indices[idx++] = curr;
                curr = parent_idx[curr];
            }
            
            // Reverse to get them in order
            for (int j = 0; j < chain_length / 2; j++) {
                int temp = chain_indices[j];
                chain_indices[j] = chain_indices[chain_length - j - 1];
                chain_indices[chain_length - j - 1] = temp;
            }
            
            // Get segment bounds
            int q_start = q_s_arr[chain_indices[0]];
            int q_end = q_e_arr[chain_indices[chain_length - 1]];
            
            int r_start = r_s_arr[chain_indices[0]];
            int r_end = r_e_arr[chain_indices[0]];
            
            for (int j = 1; j < chain_length; j++) {
                int idx = chain_indices[j];
                if (r_s_arr[idx] < r_start) r_start = r_s_arr[idx];
                if (r_e_arr[idx] > r_end) r_end = r_e_arr[idx];
            }
            
            // Add segment
            segment_count++;
            segments = (Segment*)realloc(segments, segment_count * sizeof(Segment));
            segments[segment_count - 1].q_s = q_start;
            segments[segment_count - 1].q_e = q_end;
            segments[segment_count - 1].r_s = r_start;
            segments[segment_count - 1].r_e = r_end;
            segments[segment_count - 1].score = dp_score[i];
            segments[segment_count - 1].strand = strand_arr[i];
            
            free(chain_indices);
        }
    }
    
    // Sort segments by q_s
    for (int i = 0; i < segment_count - 1; i++) {
        for (int j = 0; j < segment_count - i - 1; j++) {
            if (segments[j].q_s > segments[j + 1].q_s) {
                Segment temp = segments[j];
                segments[j] = segments[j + 1];
                segments[j + 1] = temp;
            }
        }
    }
    
    // Select non-overlapping segments
    if (segment_count == 0) {
        free(q_s_arr);
        free(q_e_arr);
        free(r_s_arr);
        free(r_e_arr);
        free(strand_arr);
        free(dp_score);
        free(parent_idx);
        return strdup("");
    }
    
    int* seg_q_s_arr = (int*)malloc(segment_count * sizeof(int));
    int* seg_q_e_arr = (int*)malloc(segment_count * sizeof(int));
    int* seg_scores_arr = (int*)malloc(segment_count * sizeof(int));
    
    for (int i = 0; i < segment_count; i++) {
        seg_q_s_arr[i] = segments[i].q_s;
        seg_q_e_arr[i] = segments[i].q_e;
        seg_scores_arr[i] = segments[i].score;
    }
    
    int* dp_select_score = (int*)malloc(segment_count * sizeof(int));
    int* prev_select_idx = (int*)malloc(segment_count * sizeof(int));
    int* selected_indices = (int*)malloc(segment_count * sizeof(int));
    
    int selected_count = select_segments_kernel(
        segment_count, seg_q_s_arr, seg_q_e_arr, seg_scores_arr,
        dp_select_score, prev_select_idx, selected_indices);
    
    // Format output
    char* output = (char*)malloc(1);
    output[0] = '\0';
    
    for (int i = selected_count - 1; i >= 0; i--) {
        int idx = selected_indices[i];
        char buffer[100];
        
        // First segment doesn't need a comma prefix
        if (i == selected_count - 1) {
            sprintf(buffer, "%d,%d,%d,%d", 
                    segments[idx].q_s,
                    segments[idx].q_e,
                    segments[idx].r_s,
                    segments[idx].r_e);
        } else {
            sprintf(buffer, ",%d,%d,%d,%d", 
                    segments[idx].q_s,
                    segments[idx].q_e,
                    segments[idx].r_s,
                    segments[idx].r_e);
        }
        
        size_t current_len = strlen(output);
        size_t buffer_len = strlen(buffer);
        output = (char*)realloc(output, current_len + buffer_len + 1);
        strcat(output, buffer);
    }
    
    // Clean up
    free(q_s_arr);
    free(q_e_arr);
    free(r_s_arr);
    free(r_e_arr);
    free(strand_arr);
    free(dp_score);
    free(parent_idx);
    free(segments);
    free(seg_q_s_arr);
    free(seg_q_e_arr);
    free(seg_scores_arr);
    free(dp_select_score);
    free(prev_select_idx);
    free(selected_indices);
    
    return output;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Usage: %s <ref_file> <query_file>\n", argv[0]);
        return 1;
    }
    
    char* ref = read_sequence_from_file(argv[1]);
    char* query = read_sequence_from_file(argv[2]);
    
    if (!ref || !query) {
        if (ref) free(ref);
        if (query) free(query);
        return 1;
    }
    
    printf("Reference length: %zu\n", strlen(ref));
    printf("Query length: %zu\n", strlen(query));
    
    // Parameters
    int kmersize = 9;
    int shift = 1;
    int max_gap_param = 250;
    int max_diag_diff_param = 150;
    float overlap_factor_param = 0.5;
    int min_anchors_param = 1;
    
    // Find k-mer matches
    int num_anchors = 0;
    Anchor* anchors = seq2hashtable_multi_test(ref, query, kmersize, shift, &num_anchors);
    
    printf("Found %d k-mer matches\n", num_anchors);
    
    // Process the alignment
    char* result = process_alignment(anchors, num_anchors, max_gap_param, max_diag_diff_param,
                                  overlap_factor_param, min_anchors_param);
    
    printf("Alignment result:\n");
    format_tuples_for_display(result);
    
    // Calculate alignment score
    int score = calculate_value(result, ref, query);
    printf("Alignment score: %d\n", score);
    
    // Clean up
    free(ref);
    free(query);
    free(anchors);
    free(result);
    
    return 0;
}
