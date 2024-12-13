from utils import extract_model_words_from_title
import numpy as np
from collections import defaultdict
from itertools import combinations
from sklearn.cluster import AgglomerativeClustering

# Step 10: Multi-component Similarity Method (MSM)
#Q-gram similarity measures how similar two strings are by breaking them into overlapping substrings of length q and comparing the resulting sets
#Computes the Jaccard similarity between the two sets of q-grams.

def q_gram_similarity(s1, s2, q=3):
    if not s1 or not s2:
        return 0
    qgrams1 = set([s1[i:i+q] for i in range(len(s1)-q+1)])
    qgrams2 = set([s2[i:i+q] for i in range(len(s2)-q+1)])
    return len(qgrams1 & qgrams2) / len(qgrams1 | qgrams2)

#HSM similarity measures how many elements in two binary vectors are identical, representing the percentage of matches.
def hsm_similarity(vector1, vector2):
    matches = np.sum(np.array(vector1) == np.array(vector2))
    return matches / len(vector1)

#TMWM similarity compares the sets of words extracted from two product titles. The similarity is the intersection-over-union (Jaccard similarity) of the word sets.
def tmwm_similarity(title1, title2):
    words1 = set(extract_model_words_from_title(title1))
    words2 = set(extract_model_words_from_title(title2))
    return len(words1 & words2) / len(words1 | words2)

def calculate_similarity(product1, product2):
    title1, title2 = product1['title'], product2['title']
    vector1 = product1.drop(['title']).values
    vector2 = product2.drop(['title']).values

    # Part 1: q-gram similarity for matching key-value pairs
    qgram_sim = q_gram_similarity(title1, title2)

    # Part 2: HSM similarity for non-matching key-value pairs
    hsm_sim = hsm_similarity(vector1, vector2)

    # Part 3: TMWM similarity for titles
    tmwm_sim = tmwm_similarity(title1, title2)

    # Combine similarities with weights
    weight_qgram, weight_hsm, weight_tmwm = 0.4, 0.2, 0.4  # Example weights
    overall_similarity = (weight_qgram * qgram_sim +
                        weight_hsm * hsm_sim +
                        weight_tmwm * tmwm_sim)
    return overall_similarity


def get_linkage_and_clusters(df, candidate_pairs):
    # Build the dissimilarity matrix
    num_products = df.shape[0]
    dissimilarity_matrix = np.ones((num_products, num_products)) * 1e6
    np.fill_diagonal(dissimilarity_matrix, 0)  # Set diagonal to 0

    for (i, j) in candidate_pairs:
        product1 = df.iloc[i]
        product2 = df.iloc[j]
        similarity = calculate_similarity(product1, product2)
        dissimilarity = 1 - similarity
        dissimilarity_matrix[i, j] = dissimilarity
        dissimilarity_matrix[j, i] = dissimilarity

    model = AgglomerativeClustering(distance_threshold=0.7,
                                    n_clusters=None,
                                    linkage='average',
                                    metric='precomputed')
    model.fit(dissimilarity_matrix)

    return model

def get_msm_candidates(df, candidate_pairs):
    model = get_linkage_and_clusters(df, candidate_pairs)
    msm_candidates = []
    for i, j in candidate_pairs:
        if model.labels_[i] == model.labels_[j]:
            msm_candidates.append((i, j))
    return msm_candidates