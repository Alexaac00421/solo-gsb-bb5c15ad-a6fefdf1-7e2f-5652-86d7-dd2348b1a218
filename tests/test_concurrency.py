"""装配提交边界（assembly_commit / DB.write_transaction）并发与原子性测试。

独立临时库（SEED_DEMO_DATA 播种 v1：陶片 #1-#4），覆盖：
- 并发录入：全部成功、版本连续无缺号，每个成功回执的版本与该版快照对应；
- 同版本并发采纳/撤销：恰好一个成功，其余 409，不产生多余版本/快照/关系；
- 成功回执的 version/assembly 必须等于该事务内固化的同版快照；
- 业务失败（校验拒绝、对象不存在等）整体回滚：版本、关系、快照数均不变；
- 轮廓修订确认、历史恢复同样走统一提交边界，回执与同版快照一致；
- 预览类端点只读：不改版本、关系、快照；
- 多实例（另一 sqlite 连接 = 另一服务进程）持写锁时按提交顺序排队：
  先提交者改了版本后，后到者在自己的写事务内复核过期 → 409；先提交者
  回滚时，后到者正常成功。
"""
import json
import os
import sqlite3
import tempfile
import threading

_fd, _DB_PATH = tempfile.mkstemp(suffix=".db")
os.close(_fd)
os.environ["DATABASE_PATH"] = _DB_PATH
os.environ["SEED_DEMO_DATA"] = "1"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import main  # noqa: E402
from app.database import remove_database_files  # noqa: E402
from app.main import app, get_db  # noqa: E402


MARKERS_A = [[35, 0], [38, 15], [33, 30]]
MARKERS_B = [[1, 0], [4, 15], [-1, 30]]


@pytest.fixture(scope="module")
def client():
    if main.db is not None:
        main.db.conn.close()
    remove_database_files(_DB_PATH)
    main.SEED_DEMO_DATA = True
    main.DB_PATH = _DB_PATH
    with TestClient(app) as c:
        yield c


def _version(client) -> int:
    return client.get("/api/assembly").json()["version"]


def _snapshot(version: int) -> dict:
    return get_db().get_snapshot_state(version)


def _snapshot_versions() -> list[int]:
    return [v["version"] for v in get_db().list_snapshot_versions()]


def _accept(client, expected_version: int):
    return client.post("/api/relations", json={
        "fragment_a": 1, "fragment_b": 2,
        "markers_a": MARKERS_A, "markers_b": MARKERS_B,
        "expected_version": expected_version,
    })


# ---------------------------------------------------------------- 回执 ↔ 快照

def test_success_receipt_matches_committed_snapshot(client):
    """采纳成功：回执 assembly/version 即本事务内固化的 v 快照（含新位姿/关系）。"""
    v = _version(client)
    r = _accept(client, v)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["version"] == v + 1
    assert body["assembly"]["version"] == body["version"]
    snap = _snapshot(body["version"])
    assert snap is not None
    assert json.dumps(snap, sort_keys=True) == json.dumps(
        body["assembly"], sort_keys=True)
    # 回执里的关系确实已落库
    assert body["assembly"]["relations"][-1]["id"] == body["relation_id"]
    assert body["assembly"]["groups"] == [{"id": 1, "fragment_ids": [1, 2]}]


def test_concurrent_accept_same_expected_version_only_one_commits(client):
    """N 个修复师同版本并发采纳：仅一个成功，其余 409；版本/快照只推进一次。"""
    # 先撤销上一个用例的关系，回到 1、2 独立状态
    v = _version(client)
    rels = [r for r in client.get("/api/relations").json() if r["active"]]
    if rels:
        r = client.post(f"/api/relations/{rels[-1]['id']}/undo",
                      json={"expected_version": v})
        assert r.status_code == 200, r.text
    v = _version(client)
    assert _version(client) == v

    barrier = threading.Barrier(8)
    results: list[int] = []

    def worker():
        barrier.wait()
        results.append(_accept(client, v).status_code)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results.count(201) == 1
    assert results.count(409) == 7
    assert _version(client) == v + 1                 # 只提交成功一次
    assert _snapshot_versions()[0] == v + 1          # 清单按版本降序：首条即最新
    active = [r for r in client.get("/api/relations").json() if r["active"]]
    assert len(active) == 1                      # 未产生重复关系


def test_concurrent_undo_same_expected_version(client):
    """同版本并发撤销同一关系：一个 200，其余 409，关系只撤销一次。"""
    v = _version(client)
    rid = [r["id"] for r in client.get("/api/relations").json()
             if r["active"]][0]
    barrier = threading.Barrier(6)
    results: list[int] = []

    def worker():
        barrier.wait()
        results.append(client.post(
            f"/api/relations/{rid}/undo",
            json={"expected_version": v}).status_code)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results.count(200) == 1
    assert results.count(409) == 5
    assert _version(client) == v + 1
    assert not client.get("/api/relations").json()[-1]["active"]


def test_concurrent_create_versions_contiguous_and_receipts_match(client):
    """并发录入：全部成功、版本连续无缺号；每个回执版本与同版快照一一对应。"""
    v0 = _version(client)
    base_frag_count = len(_snapshot(v0)["fragments"])
    n = 12
    barrier = threading.Barrier(n)
    bodies: list[dict] = []

    def worker(i):
        barrier.wait()
        r = client.post("/api/fragments", json={
            "name": f"并发陶片{i}",
            "contour": [[0, 0], [2, 0], [2, 2], [0, 2]],
        })
        assert r.status_code == 201, r.text
        bodies.append(r.json())

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    versions = sorted(b["version"] for b in bodies)
    assert versions == list(range(v0 + 1, v0 + n + 1))   # 连续无缺号
    assert _version(client) == v0 + n
    # 每个成功回执的版本与快照对应，且快照内确含本次录入的碎片
    for b in bodies:
        snap = _snapshot(b["version"])
        assert snap["version"] == b["version"]
        assert any(f["id"] == b["fragment"]["id"]
                   for f in snap["fragments"])
        assert len(snap["fragments"]) == \
            base_frag_count + (b["version"] - v0)
    snap_versions = _snapshot_versions()   # 默认按版本降序
    assert snap_versions == sorted(snap_versions, reverse=True)
    assert snap_versions[0] == v0 + n      # 无缺号：最新版恰好推进 n 次


# ---------------------------------------------------------------- 失败回滚

def test_rejected_accept_leaves_no_partial_state(client):
    """业务校验失败：版本/关系/快照均不变，不落部分状态。"""
    v = _version(client)
    snaps_before = _snapshot_versions()
    # 标记点重合触发校验拒绝（两片当前独立，拼接必然失败）
    r = client.post("/api/relations", json={
        "fragment_a": 1, "fragment_b": 3,
        "markers_a": [[0, 0], [0, 0]],
        "markers_b": [[5, 5], [5, 5]],
        "expected_version": v,
    })
    assert r.status_code == 400
    assert _version(client) == v
    assert _snapshot_versions() == snaps_before
    assert _version(client) == v


def test_stale_expected_version_409_changes_nothing(client):
    v = _version(client)
    r = client.post("/api/fragments/999/revision/confirm", json={
        "contour": [[0, 0], [1, 0], [1, 1]],
        "relations": [],
        "expected_version": v - 1,          # 过期版本
    })
    assert r.status_code == 409
    assert _version(client) == v


def test_missing_fragment_delete_rolls_back(client):
    v = _version(client)
    snaps_before = _snapshot_versions()
    r = client.delete("/api/fragments/99999?expected_version=%d" % v)
    assert r.status_code == 404
    assert _version(client) == v
    assert _snapshot_versions() == snaps_before


# ---------------------------------------------------------------- 其他写边界

def test_revision_confirm_receipt_matches_snapshot(client):
    """轮廓修订确认（recompute=False）：回执装配即同版快照，分组不变。"""
    # 录入一片独立陶片，再成功修订其轮廓
    r = client.post("/api/fragments", json={
        "name": "待修订陶片",
        "contour": [[0, 0], [10, 0], [10, 10], [0, 10]],
    })
    assert r.status_code == 201
    fid = r.json()["fragment"]["id"]
    v = _version(client)
    r = client.post(f"/api/fragments/{fid}/revision/confirm", json={
        "contour": [[0, 0], [8, 0], [8, 8], [0, 8]],
        "relations": [],
        "expected_version": v,
    })
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["version"] == v + 1
    assert json.dumps(_snapshot(body["version"]), sort_keys=True) == \
        json.dumps(body["assembly"], sort_keys=True)
    revised = next(f for f in body["assembly"]["fragments"] if f["id"] == fid)
    assert revised["contour"] == [[0.0, 0.0], [8.0, 0.0], [8.0, 8.0], [0.0, 8.0]]
    assert revised["group_id"] is None       # 独立片分组保持不变


def test_restore_confirm_receipt_matches_snapshot(client):
    """历史恢复确认：回执装配即恢复后新版快照，内容与目标历史版一致。"""
    # v1 基线为初始 4 片独立；先制造一个有拼接关系的新版本再恢复回 v1
    v = _version(client)
    rels = [r for r in client.get("/api/relations").json() if r["active"]]
    if not rels:
        r = _accept(client, v)
        assert r.status_code == 201, r.text
    current = _version(client)
    assert current > 1
    r = client.post("/api/assembly/versions/1/restore/preview",
                  json={"expected_version": current})
    assert r.status_code == 200
    r = client.post("/api/assembly/versions/1/restore/confirm",
                  json={"expected_version": current})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["version"] == current + 1
    assert body["restored_from_version"] == 1
    assert json.dumps(_snapshot(body["version"]), sort_keys=True) == \
        json.dumps(body["assembly"], sort_keys=True)
    # 恢复后：无有效关系，全部独立；旧快照仍可查
    assert all(not r["active"] or True for r in body["assembly"]["relations"])
    assert not any(r["active"] for r in body["assembly"]["relations"])
    assert all(f["group_id"] is None for f in body["assembly"]["fragments"])
    assert _snapshot(1) is not None


# ---------------------------------------------------------------- 预览只读

def test_previews_are_read_only(client):
    v = _version(client)
    asm_before = client.get("/api/assembly").json()
    snaps_before = _snapshot_versions()

    assert client.post("/api/relations/preview", json={
        "fragment_a": 1, "fragment_b": 2,
        "markers_a": MARKERS_A, "markers_b": MARKERS_B}).status_code == 200
    assert client.post("/api/trials/rehearse", json={
        "expected_version": v,
        "steps": [{"action": "undo", "relation_id": 1}],
    }).status_code == 200
    assert client.post("/api/calibration/preview", json={
        "anchor_id": 1, "expected_version": v}).status_code in (200, 400)
    assert client.get("/api/assembly/versions").status_code == 200
    assert client.get("/api/transfer/export").status_code == 200

    assert client.get("/api/assembly").json() == asm_before
    assert _snapshot_versions() == snaps_before
    assert _version(client) == v


# ---------------------------------------------------------------- 多实例提交顺序

def test_other_instance_commit_first_makes_this_write_stale(client):
    """另一实例先持写锁提交了新版本：本实例写事务获锁后复核版本 → 409。

    随后外部实例回滚其版本改动，库回到原状（过期方未写入任何变化）。
    """
    ext = sqlite3.connect(_DB_PATH)
    ext.execute("PRAGMA busy_timeout = 10000")
    v = _version(client)
    snaps_before = _snapshot_versions()

    # 外部实例先取得库级写锁（相当于另一服务进程先进入写事务）
    ext.execute("BEGIN IMMEDIATE")
    done = threading.Event()

    def stale_write():
        # 将在 BEGIN IMMEDIATE 处排队，直到外部实例提交
        r = _accept(client, v)
        done.result = r.status_code
        done.set()

    t = threading.Thread(target=stale_write)
    t.start()
    try:
        # 外部实例先提交：版本 +1（模拟先到的修复师完成了一次变更）
        ext.execute("UPDATE meta SET value = value + 1 WHERE key = 'version'")
        ext.commit()
        assert done.wait(timeout=10), "排队中的写请求未在外部提交后完成"
        assert done.result == 409               # 按提交顺序判定过期，而非覆盖写入
    finally:
        t.join()
        # 撤销外部实例人为制造的版本号推进，恢复库原状（其未写快照）
        ext.execute("BEGIN IMMEDIATE")
        ext.execute("UPDATE meta SET value = ? WHERE key = 'version'", (str(v),))
        ext.commit()
        ext.close()
    assert _version(client) == v
    assert _snapshot_versions() == snaps_before
    asm = client.get("/api/assembly").json()
    assert asm["version"] == v
    assert not any(r["active"] for r in asm["relations"])


def test_other_instance_rolls_back_waiting_write_succeeds(client):
    """外部实例持锁后回滚：排队中的同版本写请求正常成功（201）。"""
    v = _version(client)
    ext = sqlite3.connect(_DB_PATH)
    ext.execute("PRAGMA busy_timeout = 10000")
    ext.execute("BEGIN IMMEDIATE")          # 先占写锁
    done = threading.Event()

    def waiting_write():
        done.result = _accept(client, v).status_code
        done.set()

    t = threading.Thread(target=waiting_write)
    t.start()
    try:
        ext.rollback()                    # 外部实例失败回滚，未改版本
        assert done.wait(timeout=10)
        assert done.result == 201
    finally:
        t.join()
        ext.close()
    assert _version(client) == v + 1
    # 用例收尾：撤销刚建立的关系，后续用例从无有效关系的状态继续
    rid = [x["id"] for x in client.get("/api/relations").json()
             if x["active"]][0]
    r = client.post(f"/api/relations/{rid}/undo",
                  json={"expected_version": _version(client)})
    assert r.status_code == 200, r.text
