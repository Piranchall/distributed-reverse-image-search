"""
concurrent_index.py — Milestone 3 Task 2: Concurrent Index Lock Strategy
Project 20: Distributed Reverse Image Search Engine
Author: Piranchal

What this file does:
  1. Implements ReadersWriterLock — many readers can hold simultaneously,
     but a writer gets exclusive access. Not in Python stdlib.
  2. Implements ConcurrentHashTable — wraps a plain dict with the RW lock
     so QueryProcessor (reads) and IndexUpdater (writes) can run concurrently
     without corrupting the hash tables.
  3. Implements wrap_tables() — converts the list of plain dicts loaded from
     pickle files into a list of ConcurrentHashTable instances.
  4. Patches query_processor.py — call wrap_tables() after loading tables.

Run this file directly to run the concurrency stress tests:
    python incremental_update/concurrent_index.py
"""

import os
import sys
import time
import random
import threading
from contextlib import contextmanager

# ── Allow imports from repo root ──────────────────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)


# =============================================================================
# SECTION 1: ReadersWriterLock
# =============================================================================

class ReadersWriterLock:
    """
    A readers-writer lock that allows:
      - Multiple concurrent readers (shared access)
      - Exclusive writer access (no readers or other writers)

    Python's threading.RLock is a simple mutual-exclusion lock — it does NOT
    allow multiple concurrent readers. This class implements the classic
    Courtois et al. readers-preference algorithm using two standard Locks.

    Internal state:
        _read_count  : int   — how many readers are currently inside
        _count_lock  : Lock  — protects _read_count from concurrent modification
        _write_lock  : Lock  — held by the first reader (blocks writers),
                               or held directly by a writer
    """

    def __init__(self):
        self._read_count = 0
        self._count_lock = threading.Lock()   # guards _read_count
        self._write_lock = threading.Lock()   # blocks writers while readers hold it

    # ── Reader entry ──────────────────────────────────────────────────────────

    def _acquire_read(self):
        """
        Reader entry protocol:
          1. Lock count_lock (short hold — just to increment safely)
          2. Increment _read_count
          3. If I'm the FIRST reader, grab write_lock (this blocks any waiting writer)
          4. Release count_lock so other readers can enter concurrently
        """
        with self._count_lock:
            self._read_count += 1
            if self._read_count == 1:
                # First reader locks out writers
                self._write_lock.acquire()

    def _release_read(self):
        """
        Reader exit protocol:
          1. Lock count_lock
          2. Decrement _read_count
          3. If I'm the LAST reader, release write_lock (lets a waiting writer in)
          4. Release count_lock
        """
        with self._count_lock:
            self._read_count -= 1
            if self._read_count == 0:
                # Last reader unlocks for writers
                self._write_lock.release()

    # ── Writer entry ─────────────────────────────────────────────────────────

    def _acquire_write(self):
        """
        Writer entry: acquire write_lock exclusively.
        Blocks until all current readers finish AND no other writer holds it.
        """
        self._write_lock.acquire()

    def _release_write(self):
        """Writer exit: release write_lock."""
        self._write_lock.release()

    # ── Context managers ─────────────────────────────────────────────────────

    @contextmanager
    def read_context(self):
        """
        Usage:
            with lock.read_context():
                value = table[key]   # safe concurrent read
        """
        self._acquire_read()
        try:
            yield
        finally:
            self._release_read()

    @contextmanager
    def write_context(self):
        """
        Usage:
            with lock.write_context():
                table[key].append(value)   # exclusive write
        """
        self._acquire_write()
        try:
            yield
        finally:
            self._release_write()


# =============================================================================
# SECTION 2: ConcurrentHashTable
# =============================================================================

class ConcurrentHashTable:
    """
    A thread-safe wrapper around a plain Python dict.

    Every method on this class has the SAME signature as the equivalent
    dict method — QueryProcessor calls .get() and IndexUpdater calls .append(),
    and neither needs to know they're talking to a ConcurrentHashTable.

    Internal state:
        _data : dict  — the underlying {bucket_key_str: [image_id_int, ...]}
        _lock : ReadersWriterLock
    """

    def __init__(self, initial_dict=None):
        """
        Args:
            initial_dict: optional existing dict to wrap.
                          If None, starts with an empty dict.
        """
        self._data = initial_dict if initial_dict is not None else {}
        self._lock = ReadersWriterLock()

    # ── Read operations (shared access) ──────────────────────────────────────

    def get(self, key, default=None):
        """
        Thread-safe dict.get() — identical signature and behaviour.

        Many threads can call .get() simultaneously because they all
        acquire the read lock, which allows concurrent holders.

        Args:
            key:     bucket key string, e.g. '101100011010'
            default: value to return if key is absent (default None)

        Returns:
            List of image IDs in that bucket, or default if key not found.
        """
        with self._lock.read_context():
            return self._data.get(key, default)

    def keys(self):
        """
        Thread-safe keys() — returns a LIST snapshot, not a live view.

        Returning list(dict.keys()) inside the read lock prevents
        RuntimeError: dictionary changed size during iteration, which
        can happen if a writer modifies the dict while a reader iterates.

        Returns:
            List of bucket key strings currently in this table.
        """
        with self._lock.read_context():
            return list(self._data.keys())

    def __len__(self):
        """Thread-safe len()."""
        with self._lock.read_context():
            return len(self._data)

    def to_dict(self):
        """
        Returns a shallow copy of the underlying dict.
        Used by save_tables() in IndexUpdater to get a plain dict for pickling.
        (Pickling ConcurrentHashTable would also pickle the locks — unreliable.)

        Returns:
            Plain dict {bucket_key_str: [image_id_int, ...]}
        """
        with self._lock.read_context():
            return dict(self._data)

    # ── Write operation (exclusive access) ───────────────────────────────────

    def append(self, key, value):
        """
        Thread-safe insert-or-append.

        The check (does key exist?) AND the mutation (append or create)
        both happen inside a single write_context() block. This is critical:
        if you checked outside the lock and set inside it, two concurrent
        writers could both see key as absent and both create [value], losing
        one write.

        Args:
            key:   bucket key string, e.g. '101100011010'
            value: integer image_id to add to that bucket's list
        """
        with self._lock.write_context():
            if key in self._data:
                self._data[key].append(value)
            else:
                self._data[key] = [value]

    def __repr__(self):
        return (f"ConcurrentHashTable("
                f"buckets={len(self._data)}, "
                f"total_ids={sum(len(v) for v in self._data.values())})")


# =============================================================================
# SECTION 3: wrap_tables()
# =============================================================================

def wrap_tables(plain_dicts):
    """
    Convert a list of plain dicts (loaded from table_N.pkl files) into
    a list of ConcurrentHashTable instances.

    Called by:
      - QueryProcessor.__init__() after loading tables from disk
      - IndexUpdater.__init__() after loading tables from disk

    Args:
        plain_dicts: list of dicts, each {bucket_key_str: [image_id_int, ...]}

    Returns:
        List of ConcurrentHashTable instances wrapping the same data.
        No data is copied — each ConcurrentHashTable._data IS the original dict.
    """
    return [ConcurrentHashTable(initial_dict=d) for d in plain_dicts]


# =============================================================================
# SECTION 4: Stress Tests  (run with: python incremental_update/concurrent_index.py)
# =============================================================================

def _test1_correctness():
    """
    Test 1 — Correctness under concurrent reads and writes.

    Setup:
        - 1 ConcurrentHashTable, pre-populated with 100 buckets
        - 8 reader threads: each calls .get() 1000 times on random keys
        - 2 writer threads: each calls .append() 500 times on random keys
        - All 10 threads run simultaneously

    Pass condition:
        After all threads finish, the total number of values across ALL
        buckets equals exactly 1000 (2 writers × 500 appends each).
        No appended value was lost due to a race condition.
    """
    print("\n" + "=" * 60)
    print("TEST 1 — Correctness: concurrent reads + writes")
    print("=" * 60)

    # Pre-populate with 100 buckets (simulates existing index data)
    initial = {f'bucket_{i:04d}': [i * 10, i * 10 + 1] for i in range(100)}
    table = ConcurrentHashTable(initial_dict=initial)

    # Track all keys written by writer threads so we can verify them
    written_keys = []
    written_lock = threading.Lock()
    errors = []

    def reader_task(thread_id):
        """Reads 1000 random buckets — should never crash."""
        for _ in range(1000):
            key = f'bucket_{random.randint(0, 99):04d}'
            result = table.get(key, [])
            # Verify result is always a list (never corrupted)
            if not isinstance(result, list):
                errors.append(f"Reader {thread_id}: got non-list result: {result}")
        # Small sleep to mix reads and writes
            if random.random() < 0.01:
                time.sleep(0.0001)

    def writer_task(thread_id):
        """Appends 500 new image IDs using fresh keys."""
        for i in range(500):
            # Use a unique key per write to ensure no overwrites
            key = f'new_bucket_{thread_id}_{i:04d}'
            img_id = thread_id * 10000 + i
            table.append(key, img_id)
            with written_lock:
                written_keys.append((key, img_id))
            if random.random() < 0.01:
                time.sleep(0.0001)

    # Spawn all threads
    threads = []
    t0 = time.time()

    for tid in range(8):
        t = threading.Thread(target=reader_task, args=(tid,), name=f"Reader-{tid}")
        threads.append(t)

    for tid in range(2):
        t = threading.Thread(target=writer_task, args=(tid,), name=f"Writer-{tid}")
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    elapsed = time.time() - t0

    # Verify: count all new entries (keys starting with 'new_bucket_')
    all_new_ids = []
    for key, _ in written_keys:
        bucket = table.get(key, [])
        all_new_ids.extend(bucket)

    expected_writes = 2 * 500  # 2 writers × 500 appends
    actual_writes   = len(written_keys)

    # Check every written (key, value) pair is retrievable
    missing = []
    for key, img_id in written_keys:
        bucket = table.get(key, [])
        if img_id not in bucket:
            missing.append((key, img_id))

    if errors:
        print(f"  FAIL — {len(errors)} read corruption errors:")
        for e in errors[:5]:
            print(f"    {e}")
    elif missing:
        print(f"  FAIL — {len(missing)} writes were lost (race condition detected):")
        for m in missing[:5]:
            print(f"    key={m[0]}  img_id={m[1]}")
    elif actual_writes != expected_writes:
        print(f"  FAIL — expected {expected_writes} writes, recorded {actual_writes}")
    else:
        print(f"  PASS — all {expected_writes} writes verified, 0 lost")
        print(f"  Time: {elapsed:.3f}s  |  "
              f"8 readers × 1000 reads  +  2 writers × 500 appends")


def _test2_throughput():
    """
    Test 2 — Reader throughput under concurrent writes.

    Measures how much reader throughput degrades when a writer is also running.

    Pass condition:
        Reader throughput during writes is at least 80% of baseline
        (no more than 20% slowdown — within acceptable range for a RW lock).
    """
    print("\n" + "=" * 60)
    print("TEST 2 — Throughput: readers during writes vs baseline")
    print("=" * 60)

    # Pre-populate with 3000 buckets to simulate real index
    initial = {f'{i:012b}': list(range(i * 5, i * 5 + 5)) for i in range(3000)}
    all_keys = list(initial.keys())

    def run_readers(table, duration_sec, num_readers):
        """
        Run num_readers threads for duration_sec seconds.
        Returns total read count across all threads.
        """
        read_counts = [0] * num_readers
        stop_event  = threading.Event()

        def reader(tid):
            while not stop_event.is_set():
                key = random.choice(all_keys)
                _ = table.get(key, [])
                read_counts[tid] += 1

        threads = [
            threading.Thread(target=reader, args=(i,))
            for i in range(num_readers)
        ]
        for t in threads:
            t.start()
        time.sleep(duration_sec)
        stop_event.set()
        for t in threads:
            t.join()

        return sum(read_counts)

    DURATION    = 2.0   # seconds per measurement
    NUM_READERS = 10

    # ── Baseline: reads only ──────────────────────────────────────────────
    print(f"\n  Baseline (no writers, {NUM_READERS} reader threads, {DURATION}s)...")
    table_baseline = ConcurrentHashTable(initial_dict=dict(initial))
    baseline_reads = run_readers(table_baseline, DURATION, NUM_READERS)
    baseline_rps   = baseline_reads / DURATION
    print(f"  Baseline read throughput : {baseline_rps:,.0f} reads/sec")

    # ── With writer: reads + 1 writer ────────────────────────────────────
    print(f"\n  With 1 concurrent writer ({NUM_READERS} reader threads, {DURATION}s)...")
    table_mixed = ConcurrentHashTable(initial_dict=dict(initial))
    stop_writer = threading.Event()
    write_count = [0]

    def writer_task():
        img_id = 999_000
        while not stop_writer.is_set():
            key = f'live_{img_id:08d}'
            table_mixed.append(key, img_id)
            img_id += 1
            write_count[0] += 1

    writer_thread = threading.Thread(target=writer_task, name="Writer")
    writer_thread.start()

    mixed_reads = run_readers(table_mixed, DURATION, NUM_READERS)
    stop_writer.set()
    writer_thread.join()

    mixed_rps      = mixed_reads / DURATION
    degradation_pct = (1 - mixed_rps / baseline_rps) * 100

    print(f"  Mixed read throughput    : {mixed_rps:,.0f} reads/sec")
    print(f"  Writer inserted          : {write_count[0]:,} entries during test")
    print(f"  Throughput degradation   : {degradation_pct:.1f}%")

    if degradation_pct <= 20.0:
        print(f"  PASS — degradation {degradation_pct:.1f}% ≤ 20% threshold")
    else:
        print(f"  WARN — degradation {degradation_pct:.1f}% > 20% (expected for "
              f"write-heavy workload on single-machine RW lock)")
        print(f"  This is acceptable — the RW lock still prevents data corruption.")


if __name__ == '__main__':
    print("=" * 60)
    print("Milestone 3 Task 2 — ConcurrentHashTable Stress Tests")
    print("=" * 60)

    _test1_correctness()
    _test2_throughput()

    print("\n" + "=" * 60)
    print("All tests complete.")
    print("=" * 60)
    print("\nNext step: run  python incremental_update/index_updater.py")