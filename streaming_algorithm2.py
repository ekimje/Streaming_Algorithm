from ucimlrepo import fetch_ucirepo
import hashlib
import random
import statistics
import sys
import time
import pandas as pd
from array import array
from collections import Counter
  
# fetch dataset 
online_retail = fetch_ucirepo(id=352) 

online = online_retail.data.features
online.to_csv("online_retail.csv", index=False)

class BloomFilter:

    def __init__(self, bit_size, hash_count):
        self.bit_size = bit_size
        self.hash_count = hash_count
        self.bits = bytearray((bit_size + 7) // 8)

    def _hashes(self, item):

        item = item.encode("utf-8")

        for seed in range(self.hash_count):

            digest = hashlib.blake2b(
                item,
                digest_size=8,
                person=seed.to_bytes(4, "little")
            ).digest()

            yield int.from_bytes(digest, "little") % self.bit_size

    def add(self, item):

        for index in self._hashes(item):
            self.bits[index // 8] |= (1 << (index % 8))

    def contains(self, item):

        for index in self._hashes(item):

            if not (self.bits[index // 8] &
                    (1 << (index % 8))):
                return False

        return True

    def memory_usage(self):
        return sys.getsizeof(self.bits)


class CountMinSketch:

    def __init__(self, width, depth):

        self.width = width
        self.depth = depth

        self.table = [
            array("I", [0] * width)
            for _ in range(depth)
        ]

    def _hashes(self, item):

        item = item.encode("utf-8")

        for seed in range(self.depth):

            digest = hashlib.blake2b(
                item,
                digest_size=8,
                person=(1000 + seed).to_bytes(4, "little")
            ).digest()

            yield int.from_bytes(
                digest,
                "little"
            ) % self.width

    def add(self, item, count=1):

        for row, col in enumerate(self._hashes(item)):
            self.table[row][col] += count

    def estimate(self, item):

        return min(
            self.table[row][col]
            for row, col in enumerate(self._hashes(item))
        )

    def memory_usage(self):

        return (
            sys.getsizeof(self.table)
            + sum(sys.getsizeof(row)
                  for row in self.table)
        )

def stream_online_retail():
    for chunk in pd.read_csv("online_retail.csv", chunksize=10000):
        
        for _, row in chunk.iterrows():
            if row["CustomerID"] != row["CustomerID"]:  # Check for NaN values
                continue
            customer_id = row["CustomerID"]
            product = row["Description"]
        
            yield customer_id, product


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = min(len(values) - 1, max(0, round((len(values) - 1) * p)))
    return values[index]


def run_experiment():

    bloom = BloomFilter(
        bit_size=1000000,
        hash_count=5
    )

    cms = CountMinSketch(
        width=2000,
        depth=5
    )

    exact_edges = set()
    exact_frequency = Counter()

    start = time.perf_counter()

    total_records = 0

    for customer, product in stream_online_retail():

        key = f"{customer}->{product}"

        bloom.add(key)

        cms.add(product)

        exact_edges.add(key)

        exact_frequency[product] += 1

        total_records += 1

    elapsed = time.perf_counter() - start

    print("레코드 수:", total_records)
    print("처리 시간:", elapsed)

    # Bloom Filter 정확도 평가
    positives = random.sample(
        list(exact_edges),
        min(10000, len(exact_edges))
    )

    negatives = [
        f"fake{i}->fake{i}"
        for i in range(len(positives))
    ]

    false_positive = 0

    for item in negatives:

        if bloom.contains(item):
            false_positive += 1

    fpr = false_positive / len(negatives)

    print("False Positive Rate =", fpr)
    
    #Count-Min Sketch 정확도 평가
    errors = []

    items = random.sample(
        list(exact_frequency.keys()),
        min(5000, len(exact_frequency))
    )

    for item in items:

        real = exact_frequency[item]

        estimate = cms.estimate(item)

        errors.append(
            (estimate - real) / real
        )

    mean_error = statistics.mean(errors)

    print("Mean Relative Error =", mean_error)
    
    # 메모리 출력
    print("Bloom Filter Memory Usage =", bloom.memory_usage(), "bytes")
    print("Count-Min Sketch Memory Usage =", cms.memory_usage(), "bytes")


if __name__ == "__main__":
    run_experiment()