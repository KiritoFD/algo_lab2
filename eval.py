# %%

import numpy as np
from numba import njit
import edlib


def get_rc(s):
    map_dict = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
    l = []
    for c in s:
        l.append(map_dict[c])
    l = l[::-1]
    return ''.join(l)
def rc(s):
    map_dict = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
    l = []
    for c in s:
        l.append(map_dict[c])
    l = l[::-1]
    return ''.join(l)

def seq2hashtable_multi_test(refseq, testseq, kmersize=15, shift = 1):
    rc_testseq = get_rc(testseq)
    testseq_len = len(testseq)
    local_lookuptable = dict()
    skiphash = hash('N'*kmersize)
    for iloc in range(0, len(refseq) - kmersize + 1, 1):
        hashedkmer = hash(refseq[iloc:iloc+kmersize])
        if(skiphash == hashedkmer):
            continue
        if(hashedkmer in local_lookuptable):

            local_lookuptable[hashedkmer].append(iloc)
        else:
            local_lookuptable[hashedkmer] = [iloc]
    iloc = -1
    readend = testseq_len-kmersize+1
    one_mapinfo = []
    preiloc = 0
    while(True):
   
        iloc += shift
        if(iloc >= readend):
            break

        #if(hash(testseq[iloc: iloc + kmersize]) == hash(rc_testseq[-(iloc + kmersize): -iloc])):
            #continue
 
        hashedkmer = hash(testseq[iloc: iloc + kmersize])
        if(hashedkmer in local_lookuptable):

            for refloc in local_lookuptable[hashedkmer]:

                one_mapinfo.append((iloc, refloc, 1, kmersize))



        hashedkmer = hash(rc_testseq[-(iloc + kmersize): -iloc])
        if(hashedkmer in local_lookuptable):
            for refloc in local_lookuptable[hashedkmer]:
                one_mapinfo.append((iloc, refloc, -1, kmersize))
        preiloc = iloc

    

    return np.array(one_mapinfo)

def get_points(tuples_str):
    data = []
    num = 0
    # Ensure tuples_str is a string, not bytes
    if isinstance(tuples_str, bytes):
        tuples_str = tuples_str.decode()
        
    for c in tuples_str:
        if '0' <= c <= '9': # Simplified check for digits
            num = num * 10 + int(c)
        elif c == ',':
            data.append(num)
            num = 0
    if len(tuples_str) > 0 : # Append last number if string was not empty and num is pending
        data.append(num)
    return data

def calculate_distance(ref, query, ref_st, ref_en, query_st, query_en):
    A = ref[ref_st: ref_en]
    a = query[query_st: query_en]
    _a = rc(query[query_st: query_en])
    return min(edlib.align(A, a)['editDistance'], edlib.align(A, _a)['editDistance'])

def get_first(x):
    return x[0]


def calculate_value(tuples_str, ref, query):  

    slicepoints = np.array(get_points(tuples_str)) # No .encode() needed if get_points handles string
    if(len(slicepoints) > 0 and len(slicepoints) % 4 == 0):
        editdistance = 0
        aligned = 0
        preend = 0
        points = np.array(slicepoints).reshape((-1, 4)).tolist()
        points.sort(key=get_first)
        for onetuple in points:
            query_st, query_en, ref_st, ref_en = onetuple
            if(preend > query_st):
                return 0
            if(query_en - query_st < 30):
                continue
            preend = query_en
            if((calculate_distance(ref, query, ref_st, ref_en, query_st, query_en)/len(query[query_st:query_en])) > 0.1):
                continue
            editdistance += calculate_distance(ref, query, ref_st, ref_en, query_st, query_en)
            aligned += len(query[query_st:query_en])
        return max(aligned - editdistance, 0)
    else:
        return 0

def format_tuples_for_display(tuples_str):
    points_list = get_points(tuples_str)
    if not points_list or len(points_list) % 4 != 0:
        return []
    
    formatted_list = []
    for i in range(0, len(points_list), 4):
        formatted_list.append(tuple(points_list[i:i+4]))
    return formatted_list

# %%
#实验1
from data import ref1,que1
# ref = ref1 # No longer needed as global
# query = que1 # No longer needed as global


# %%
# data = seq2hashtable_multi_test(ref, query, kmersize=9, shift = 1) # Processed per dataset
# data.shape # Processed per dataset

# %%
#实验2
from data import ref2, que2
# ref = ref2 # No longer needed as global
# query = que2 # No longer needed as global


# %%
# data = seq2hashtable_multi_test(ref, query, kmersize=9, shift = 1) # Processed per dataset
# data.shape # Processed per dataset

# %%


# %% [markdown]
# 在这里设计你的算法

# %%
#Design a algorithm
from run import function
# %%


# %% [markdown]
# Result

# %%
# Process Dataset 1
print("Processing Dataset 1...")
data1 = seq2hashtable_multi_test(ref1, que1, kmersize=9, shift=1)
print(f"Dataset 1 k-mer matches shape: {data1.shape}")
tuples_str1 = str(function(data1))
print("Alignment Result for Dataset 1:")
formatted_output1 = format_tuples_for_display(tuples_str1)
print(formatted_output1)
score1 = calculate_value(tuples_str1, ref1, que1)
print(f"Final Score for Dataset 1: {score1}\n")

# Process Dataset 2
print("Processing Dataset 2...")
data2 = seq2hashtable_multi_test(ref2, que2, kmersize=9, shift=1)
print(f"Dataset 2 k-mer matches shape: {data2.shape}")
tuples_str2 = str(function(data2))
print("Alignment Result for Dataset 2:")
formatted_output2 = format_tuples_for_display(tuples_str2)
print(formatted_output2)
score2 = calculate_value(tuples_str2, ref2, que2)
print(f"Final Score for Dataset 2: {score2}")

# %%
#Score
# score = calculate_value(tuples_str, ref, query) # Now done per dataset
# print(f"Final Score: {score}") # Now done per dataset

# %%



