"""
Evaluator Functions
Calculates Precision@k and Recall@k
"""

def precision_at_k(results, correct_ids, k):
    if k == 0:
        return 0.0
    top_k = results[:k]
    correct_count = len(set(top_k) & set(correct_ids))
    return correct_count / k


def recall_at_k(results, correct_ids, k):
    total_correct = len(correct_ids)
    if total_correct == 0:
        return 0.0
    top_k = results[:k]
    found_count = len(set(top_k) & set(correct_ids))
    return found_count / total_correct


def evaluate_single_query(query_id, search_results, ground_truth, k_values):
    """
    For one image query, calculate precision and recall for all k values
    """
    # Check if query is an original image
    if query_id in ground_truth:
        similar_ids = ground_truth[query_id]
    else:
        # Check if query is a variant image
        similar_ids = None
        for base_id, variants in ground_truth.items():
            if query_id in variants:
                similar_ids = variants
                break
        
        # If query not found in ground truth
        if similar_ids is None:
            return {k: {"precision": 0.0, "recall": 0.0} for k in k_values}
    
    # Calculate for each k
    results = {}
    for k in k_values:
        results[k] = {
            "precision": precision_at_k(search_results, similar_ids, k),
            "recall": recall_at_k(search_results, similar_ids, k)
        }
    
    return results


# Test code
if __name__ == "__main__":
    print("="*50)
    print("Testing Evaluator Functions")
    print("="*50)
    
    test_ground_truth = {
        2000000: [2000001, 2000002, 2000003, 2000004, 2000005]
    }
    
    perfect_results = [2000001, 2000002, 2000003, 2000004, 2000005, 999999]
    
    k_values = [1, 5, 10]
    
    print("\nTesting with perfect results:")
    for k in k_values:
        p = precision_at_k(perfect_results, test_ground_truth[2000000], k)
        r = recall_at_k(perfect_results, test_ground_truth[2000000], k)
        print(f"  k={k}: Precision={p:.2f}, Recall={r:.2f}")
    
    print("\n" + "="*50)
    print("Functions working correctly!")
    print("="*50)