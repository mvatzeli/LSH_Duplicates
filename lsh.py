import numpy as np
from collections import defaultdict


class LSH:
    def __init__(self, M, thresh):
        self.num_funcs = M.shape[0]
        self.N = M.shape[1]
        self.thresh = thresh
        self.band_buckets = []
        self.b, self.r = self._get_b_r(self.num_funcs, self.thresh)
        for band_idx in range(self.b):
            bucket = defaultdict(set)
            for j in range(self.N):
                key = "|".join(map(str, M[(band_idx * self.r) : ((band_idx + 1) * self.r), j]))
                bucket.setdefault(key, set()).add(j)
            self.band_buckets.append(bucket)
    
    def _get_b_r(self, num_funcs, thresh):
        best_diff = float('inf')
        best_b, best_r = None, None
        for r in range(1, num_funcs + 1):
            if num_funcs % r == 0:
                b = num_funcs // r
                cur_t = (1 / b) ** (1 / r)
                diff = abs(cur_t - thresh)
                if diff < best_diff:
                    best_diff = diff
                    best_b = b
                    best_r = r
        return best_b, best_r

    def get_candidate_pairs(self):
        bucket_cnts = np.zeros((self.N, self.N))
        for band_map in self.band_buckets:
            for key in band_map.keys():
                if len(band_map[key]) > 1:
                    for i in band_map[key]:
                        for j in band_map[key]:
                            if i < j:
                                bucket_cnts[i, j] = 1
        return np.argwhere(bucket_cnts >= 1)
