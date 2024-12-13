import numpy as np
import pandas as pd
from utils import extract_model_words_from_title, extract_model_words_from_value


def create_binary_vectors(df):
    all_words = set()

    for column in df.columns:
        if column == "title":
            for text in df[column]:
                all_words.update(extract_model_words_from_title(text))
        else:
            for text in df[column]:
                all_words.update(extract_model_words_from_value(text))

    all_words_index = {word: i for i, word in enumerate(all_words)}

    binary_vectors = np.zeros((len(all_words), len(df)), dtype=int)

    i = 0
    for idx, row in df.iterrows():
        binary_vector = np.zeros(len(all_words), dtype=int)
        for column in df.columns:
            words = (
                extract_model_words_from_value(row[column])
                if column != "title"
                else extract_model_words_from_title(row[column])
            )
            for word in words:
                binary_vector[all_words_index[word]] = 1

        binary_vectors[:, i] = binary_vector
        i += 1

    return binary_vectors


def create_minhash(df_mh):
    binary_matrix = create_binary_vectors(df_mh)
    num_features = binary_matrix.shape[0]
    # Step 8: Apply Minhash Signatures for Locality Sensitive Hashing (LSH)

    # Define the number of minhash functions and a large prime number
    num_minhash_functions = int(num_features * 0.5)  # half the number of features
    prime_p = 11047  # A large prime number

    # Create hash functions of the form h(x) = (a*x + b) mod p
    def create_hash_function(a, b, p):
        return lambda x: (a * x + b) % p

    hash_functions = []
    for _ in range(num_minhash_functions):
        a = np.random.randint(1, prime_p)
        b = np.random.randint(0, prime_p)
        hash_functions.append(create_hash_function(a, b, prime_p))

    # Create the Minhash signature matrix
    minhash_matrix = np.full((num_minhash_functions, binary_matrix.shape[1]), np.inf)

    # Compute minhash signatures
    for i, binary_vector in enumerate(binary_matrix.T):
        for j in range(num_features):
            if binary_vector[j] == 1:
                for k, hash_function in enumerate(hash_functions):
                    minhash_matrix[k, i] = min(minhash_matrix[k, i], hash_function(j))

    # Convert Minhash matrix to integer values
    minhash_matrix = minhash_matrix.astype(int)
    return minhash_matrix
