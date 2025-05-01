
import sys
import requests
import json
from typing import List, Tuple
import logging

# Assuming the optimized align function is available
from dna import align, read_fasta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def submit_results(results: List[Tuple[int, int, int, int]], url: str, username: str = "test") -> dict:
    """
    Submit alignment results to the specified URL with the given username.
    Returns the server response as a dictionary.
    """
    try:
        # Format results as JSON
        payload = {
            "username": username,
            "results": [{"query_start": r[0], "query_end": r[1], "ref_start": r[2], "ref_end": r[3]} for r in results]
        }
        headers = {"Content-Type": "application/json"}
        
        logger.info(f"Submitting results to {url} with username {username}")
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to submit to {url}: {e}")
        return {"error": str(e), "score": 0}

def analyze_results(response: dict, group: str) -> List[str]:
    """
    Analyze the server response and suggest code modifications based on score and feedback.
    Returns a list of suggested improvements.
    """
    suggestions = []
    score = response.get("score", 0)
    feedback = response.get("feedback", "No feedback provided")
    
    logger.info(f"Group {group} - Score: {score}, Feedback: {feedback}")
    
    # Threshold for acceptable score (based on lab's baseline requirement)
    baseline_score = 80  # Adjust based on lab's baseline if known
    
    if score < baseline_score:
        suggestions.append(f"Score for group {group} ({score}) is below baseline ({baseline_score}).")
        
        # Analyze feedback for common issues
        if "inversion" in feedback.lower():
            suggestions.append("Inversion detection may be incomplete. Ensure reverse complement seeds are correctly chained and merged.")
            suggestions.append("Check if `chain_seeds` prioritizes the correct strand when forward and reverse chains have similar lengths.")
        if "translocation" in feedback.lower() or "shift" in feedback.lower():
            suggestions.append("Translocation handling may be weak. Increase `max_gap` (e.g., to 100) to allow larger segment shifts.")
            suggestions.append("Modify `merge_chain` to allow non-contiguous segments with larger gaps.")
        if "mismatch" in feedback.lower() or "snp" in feedback.lower():
            suggestions.append("SNP tolerance may be too strict. Increase `max_mismatch` (e.g., to 3 or 4) in `merge_chain`.")
            suggestions.append("Verify local alignment in `merge_chain` accounts for small mismatches correctly.")
        if "insertion" in feedback.lower() or "deletion" in feedback.lower():
            suggestions.append("Insertion/deletion handling may be insufficient. Adjust `max_gap` and `max_mismatch` to tolerate larger indels.")
        if "complexity" in feedback.lower() or score == 0:
            suggestions.append("Algorithm may be too slow. Optimize k-mer index by using a larger `k` (e.g., 15) to reduce seed count.")
            suggestions.append("Check if `chain_seeds` can skip redundant seeds using a heuristic filter.")
    
    # General suggestions for low scores
    if score < baseline_score:
        suggestions.append("Validate output format: Ensure results are in [(q_start, q_end, r_start, r_end)] as per Lab2 section 3.1.")
        suggestions.append("Test with sample data from Lab2 PPT slides (pages 4-7) to identify specific mutation handling issues.")
        suggestions.append("Consider increasing `k` (e.g., to 13 or 15) to improve seed specificity, balancing with sensitivity.")
    
    return suggestions

def main(query_file: str, ref_file: str):
    """
    Main function to run alignment, submit results, and suggest improvements.
    """
    # Read input sequences
    try:
        query_seq = read_fasta(query_file)
        ref_seq = read_fasta(ref_file)
    except Exception as e:
        logger.error(f"Failed to read input files: {e}")
        return
    
    # Run alignment
    try:
        results = align(query_seq, ref_seq, k=11, max_gap=50, max_mismatch=2)
        logger.info(f"Alignment results: {results}")
    except Exception as e:
        logger.error(f"Alignment failed: {e}")
        return
    
    # Submit to both groups
    urls = {
        "Group 1": "http://10.20.26.11:8550",
        "Group 2": "http://10.20.26.11:8551"
    }
    
    all_suggestions = []
    for group, url in urls.items():
        response = submit_results(results, url, username="test")
        suggestions = analyze_results(response, group)
        all_suggestions.extend(suggestions)
    
    # Print unique suggestions
    if all_suggestions:
        logger.info("Suggested code modifications:")
        for i, suggestion in enumerate(set(all_suggestions), 1):
            print(f"{i}. {suggestion}")
    else:
        logger.info("No modifications suggested. Scores are satisfactory.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python submit_dna_alignment.py <query.fa> <ref.fa>")
        sys.exit(1)
    
    main(sys.argv[1], sys.argv[2])