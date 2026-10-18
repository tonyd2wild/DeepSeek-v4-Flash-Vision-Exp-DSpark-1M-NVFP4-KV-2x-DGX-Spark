# SPDX-License-Identifier: Apache-2.0
"""dspark031 plugin (KV-7 dedupe, MNT-10) against vLLM 0.31's real classes."""

import inspect
import json
import os
import sys

sys.path.insert(0, os.environ.get("DSPARK031_DIR", "/p"))

import dspark031 as P  # noqa: E402
from vllm.v1.core import block_pool as bp  # noqa: E402
from vllm.v1.core.kv_cache_utils import (  # noqa: E402
    BlockHash, make_block_hash_with_group_id)
from vllm.v1.core.sched import scheduler as sm  # noqa: E402

UPSTREAM_INSERT = bp.BlockPool._insert_block_hash


def _key(n, g=0):
    return make_block_hash_with_group_id(BlockHash(bytes([n]) * 32), g)


def _pool():
    return bp.BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=16)


def test_upstream_still_duplicates():
    """The defect, on stock 0.31: two blocks end up under one key."""
    bp.BlockPool._insert_block_hash = UPSTREAM_INSERT
    p = _pool()
    a, b = p.blocks[1], p.blocks[2]
    p._insert_block_hash(_key(1), a, 16)
    p._insert_block_hash(_key(1), b, 16)
    assert p.cached_block_hash_to_block.contain(_key(1), a.block_id)
    assert p.cached_block_hash_to_block.contain(_key(1), b.block_id)


def test_dedupe_evicts_the_stale_single_hash_copy():
    assert P.install_block_dedupe()
    assert P.install_block_dedupe()            # idempotent
    p = _pool()
    a, b = p.blocks[1], p.blocks[2]
    p._insert_block_hash(_key(2), a, 16)
    before = P.STATS["dedupe_evicted"]
    p._insert_block_hash(_key(2), b, 16)
    m = p.cached_block_hash_to_block
    assert m.contain(_key(2), b.block_id) and not m.contain(_key(2), a.block_id)
    assert a.block_hash is None, "the stale copy keeps a hash"
    assert P.STATS["dedupe_evicted"] == before + 1


def test_dedupe_leaves_a_multi_hash_block_alone():
    """0.31: a block can carry several hashes, and evicting it would strip
    all of them -- so the dedupe stands aside and counts it."""
    assert P.install_block_dedupe()
    p = _pool()
    a, b = p.blocks[1], p.blocks[2]
    p._insert_block_hash(_key(3), a, 16)       # primary
    p._insert_block_hash(_key(4), a, 32)       # a second hash on the same block
    assert p.cached_block_hashes_by_block.get(a.block_id)
    skipped = P.STATS["dedupe_skipped_multi_hash"]
    p._insert_block_hash(_key(3), b, 16)
    m = p.cached_block_hash_to_block
    assert m.contain(_key(3), a.block_id) and m.contain(_key(4), a.block_id), (
        "the other hash of the old block was evicted")
    assert P.STATS["dedupe_skipped_multi_hash"] == skipped + 1


def test_dedupe_off_at_boot_is_not_installed(monkeypatch):
    """DSPARK_BLOCK_DEDUPE=0 at boot: upstream's method stays in place, so the
    boot log cannot claim KV-7 is armed when it is off."""
    bp.BlockPool._insert_block_hash = UPSTREAM_INSERT
    monkeypatch.setenv("DSPARK_BLOCK_DEDUPE", "0")
    assert P.install_block_dedupe() is False
    assert bp.BlockPool._insert_block_hash is UPSTREAM_INSERT


def test_dedupe_kill_switch(monkeypatch):
    assert P.install_block_dedupe()
    monkeypatch.setenv("DSPARK_BLOCK_DEDUPE", "0")
    p = _pool()
    a, b = p.blocks[1], p.blocks[2]
    p._insert_block_hash(_key(5), a, 16)
    p._insert_block_hash(_key(5), b, 16)
    m = p.cached_block_hash_to_block
    assert m.contain(_key(5), a.block_id) and m.contain(_key(5), b.block_id)


def test_upstream_schedule_reads_the_field_every_call():
    """The pin MNT-10 rests on: 0.31's schedule() reads the config field
    itself, so setting it before each call is exact."""
    src = inspect.getsource(UPSTREAM_SCHEDULE)
    assert "self.scheduler_config.long_prefill_token_threshold" in src


UPSTREAM_SCHEDULE = sm.Scheduler.schedule


def test_mnt10_precedence_and_wrapper(tmp_path, monkeypatch):
    flags = tmp_path / "flags.json"
    monkeypatch.setattr(P, "_FLAGS", str(flags))
    P._cache[0] = 0.0
    monkeypatch.delenv("VLLM_LONG_PREFILL_TOKEN_THRESHOLD", raising=False)
    assert P.threshold(4096) == 4096, "no override -> boot value"
    flags.write_text(json.dumps({"long_prefill_token_threshold": 1024}))
    P._cache[0] = 0.0
    assert P.threshold(4096) == 1024, "the flags file overrides"
    monkeypatch.setenv("VLLM_LONG_PREFILL_TOKEN_THRESHOLD", "512")
    assert P.threshold(4096) == 512, "the env var wins over the file"
    monkeypatch.setenv("VLLM_LONG_PREFILL_TOKEN_THRESHOLD", "-3")
    assert P.threshold(4096) == 4096, "a negative value is ignored"
    monkeypatch.delenv("VLLM_LONG_PREFILL_TOKEN_THRESHOLD")
    flags.write_text("{half-writ")
    P._cache[0] = 0.0
    assert P.threshold(4096) == 1024, "a torn file keeps the LAST GOOD value"

    # the wrapper sets the field schedule() reads, before calling it
    seen = []
    monkeypatch.setattr(sm.Scheduler, "schedule",
                        lambda self, *a, **k: seen.append(
                            self.scheduler_config.long_prefill_token_threshold))
    assert P.install_live_long_prefill()

    class Cfg:
        long_prefill_token_threshold = 4096

    s = sm.Scheduler.__new__(sm.Scheduler)
    s.scheduler_config = Cfg()
    sm.Scheduler.schedule(s)
    assert seen == [1024], seen
    assert s._dspark031_boot_lpt == 4096, "the boot value is remembered"
    flags.write_text(json.dumps({}))
    P._cache[0] = 0.0
    sm.Scheduler.schedule(s)
    assert seen == [1024, 4096], "removing the key restores the boot value"
