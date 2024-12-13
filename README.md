## LSH FOR Scalable Product Duplicate Detection

This is the code for my final project for the course of Computer Science For Business Analytics at Erasmus University of Rotterdam (ESE). Briefly, in this assignment, I work with a dataset of 1,624 descriptions of televisions coming from four Webshops: Amazon.com, Newegg.com, Best-Buy.com, and TheNerds.net and I try to find th eproduct duplicates without using the modelId of the products. In my code, I first preprocess the data by normalizing the columns and selecting relevant features that I will compare. Then, I use MinHashing and LSH to find the potential duplicates pairs and I also apply the Multi-component Similarity Method (MSM) in order to cluster the products.

### Code Structure
The code that generates the plots found in the report is located in the file `CS Assignment.ipynb`. The file `utils.py` contains helper functions related to data pre processing and evaluation. The file `minhash.py` contains all necessary functions to build the Minhash signature matrix. The file `msm.py` includes all methods needed to perform MSM. Finally, `lsh.py` contains the class that performs Locality Sensitive Hashing. Before running the code run `pip install -r requirements.txt` to install all necessary packages.

### Implementation details of LSH 
The LSH class consists of the following methos:
*  `__init__`: The initializer takes as arguments the $n \times N$ signature matrix $M$ (where $n$ is the number of hash functions and $N$ the number of binary vectors) and a similarity threshold $t$.
* `_get_b_r`: This private method returns the optimal values for $b$ and $r$ given the desired similarity threshold (for each divisor $r$ of $n$, it finds the appropriate value of $b$ as $b = n / r$ and computes the similarity threshold $t' = (1/b)^{1/r}$. It picks the values of $r$ and $b$ such that the resulting $t'$ is closest to $t$).
* `get_candidate_pairs`: It returns all candidate pairs such that with high probability their similarity is at least $t$. It does so by iterating over each bucket and returning all possible combinations of pairs within the bucket.