"""
query_cache.py — Task 6: Query Caching Layer
Project 20: Distributed Reverse Image Search Engine — Milestone 2
"""

import os, json, time, collections
import numpy as np
from PIL import Image
import imagehash
from tqdm import tqdm

_CFG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'lsh_index', 'lsh_config.json')
with open(_CFG_PATH) as _f:
    _CFG = json.load(_f)

CACHE_MAX_SIZE          = _CFG['cache_max_size']
CACHE_HAMMING_THRESHOLD = _CFG['cache_hamming_threshold']
TOP_K                   = _CFG['top_k']
HASH_SIZE               = 8


def hamming_distance(h1: int, h2: int) -> int:
    return bin(h1 ^ h2).count('1')


def hash_similarity(h1: int, h2: int) -> float:
    return 1.0 - (hamming_distance(h1, h2) / 64.0)


class LRUCache:
    def __init__(self, max_size: int):
        if max_size < 1:
            raise ValueError('max_size must be >= 1')
        self._max_size = max_size
        self._store    = collections.OrderedDict()

    def get(self, phash_key: int):
        if phash_key not in self._store:
            return None
        result_list, _ = self._store[phash_key]
        self._store.move_to_end(phash_key)
        self._store[phash_key] = (result_list, time.time())
        return result_list

    def put(self, phash_key: int, result_list: list):
        if phash_key in self._store:
            self._store.move_to_end(phash_key)
        self._store[phash_key] = (result_list, time.time())
        if len(self._store) > self._max_size:
            self._store.popitem(last=False)

    def find_approximate(self, query_phash: int, threshold: int):
        best_key, best_dist = None, threshold + 1
        for cached_key, (result_list, _) in reversed(self._store.items()):
            dist = hamming_distance(query_phash, cached_key)
            if dist < best_dist:
                best_dist, best_key = dist, cached_key
        if best_key is not None and best_dist <= threshold:
            result_list, _ = self._store[best_key]
            self._store.move_to_end(best_key)
            self._store[best_key] = (result_list, time.time())
            return result_list
        return None

    def clear(self):
        self._store.clear()

    def __len__(self):
        return len(self._store)

    def __repr__(self):
        return f'LRUCache(size={len(self)}/{self._max_size})'


class CachedSearchEngine:
    def __init__(self, search_engine, phash_lookup: dict,
                 max_size: int = CACHE_MAX_SIZE,
                 hamming_thresh: int = CACHE_HAMMING_THRESHOLD):
        self.search_engine    = search_engine
        self.phash_lookup     = phash_lookup
        self.hamming_thresh   = hamming_thresh
        self.cache            = LRUCache(max_size=max_size)
        self.total_queries    = 0
        self.cache_hits       = 0
        self.exact_hits       = 0
        self.approximate_hits = 0
        self.cache_misses     = 0
        self.cache_bypasses   = 0
        print(f'✓ CachedSearchEngine ready. Cache={max_size}, thresh={hamming_thresh}')

    def cached_search(self, query_image_id=None, query_embedding=None,
                      query_pil_image=None, top_k: int = TOP_K) -> list:
        n = sum(x is not None for x in [query_image_id, query_embedding, query_pil_image])
        if n != 1:
            raise ValueError('Provide exactly one input type.')
        self.total_queries += 1

        if query_embedding is not None:
            self.cache_bypasses += 1
            emb  = np.array(query_embedding, dtype=np.float32).flatten()
            norm = np.linalg.norm(emb)
            emb  = emb / norm if norm > 1e-8 else emb
            return self.search_engine.search(query_embedding=emb,
                                             query_image_id=None, top_k=top_k)

        if query_pil_image is not None:
            ph = int(str(imagehash.phash(query_pil_image, hash_size=HASH_SIZE)), 16)
            return self._lookup_and_search(ph,
                       lambda: self._embed_pil(query_pil_image), None, top_k)

        if query_image_id not in self.phash_lookup:
            raise KeyError(f'image_id {query_image_id} not in phash_lookup.')
        ph = self.phash_lookup[query_image_id]['phash']
        return self._lookup_and_search(ph,
                   lambda: self._get_embedding_for_id(query_image_id),
                   query_image_id, top_k)

    def _lookup_and_search(self, query_phash, embedding_fn, image_id, top_k):
        cached = self.cache.get(query_phash)
        if cached is not None:
            self.cache_hits += 1
            self.exact_hits += 1
            return cached
        approx = self.cache.find_approximate(query_phash, self.hamming_thresh)
        if approx is not None:
            self.cache_hits       += 1
            self.approximate_hits += 1
            self.cache.put(query_phash, approx)
            return approx
        self.cache_misses += 1
        result = self.search_engine.search(query_embedding=embedding_fn(),
                                           query_image_id=image_id, top_k=top_k)
        self.cache.put(query_phash, result)
        return result

    def _get_embedding_for_id(self, image_id):
        return self.search_engine.query_processor.prepare_query_embedding(
            image_id=image_id)

    def _embed_pil(self, pil_image):
        raise NotImplementedError(
            'PIL embedding extraction requires a CNN model. '
            'Use query_embedding= instead.')

    def warm_cache(self, image_ids: list, top_k: int = TOP_K):
        print(f'Warming cache with {len(image_ids)} images ...')
        for iid in tqdm(image_ids, desc='Warm-up', unit='img'):
            try:
                self.cached_search(query_image_id=iid, top_k=top_k)
            except Exception as e:
                print(f'  Skipped {iid}: {e}')

    def get_cache_stats(self) -> dict:
        hit_rate = (self.cache_hits / self.total_queries * 100.0
                    if self.total_queries > 0 else 0.0)
        try:
            mean_ms = self.search_engine.get_search_stats().get('mean_latency_ms', 0.0)
        except Exception:
            mean_ms = 0.0
        return {
            'total_queries':           self.total_queries,
            'cache_hits':              self.cache_hits,
            'cache_misses':            self.cache_misses,
            'exact_hits':              self.exact_hits,
            'approximate_hits':        self.approximate_hits,
            'cache_bypasses':          self.cache_bypasses,
            'hit_rate_pct':            round(hit_rate, 2),
            'current_cache_size':      len(self.cache),
            'max_cache_size':          self.cache._max_size,
            'mean_search_latency_ms':  round(mean_ms, 2),
            'estimated_time_saved_ms': round(self.cache_hits * mean_ms, 2),
        }

    def save_cache_stats(self, output_path2: str = None):
        if output_path2 is None:
            output_path2 = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'cache_stats.json')
        stats = self.get_cache_stats()
        with open(output_path2, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f'✓ Cache stats → {output_path2}')
        return stats
