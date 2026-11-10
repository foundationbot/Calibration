"""calibtool.storage: session folders, raw-data retention, protected paths and disk checks (design section 11)."""
import datetime as dt

import pytest

pytest.importorskip("numpy")

from calibtool import storage as st  # noqa: E402


def test_session_id():
    t = dt.datetime(2026, 11, 9, 19, 5, 7).timestamp()
    assert st.new_session_id("P10", t) == "20261109-190507-P10"


def test_session_dirs(tmp_path):
    d = st.SessionDirs(tmp_path, "20261109-190507-P10").create()
    for k in ("raw", "corners", "fit", "report", "deploy"):
        assert (tmp_path / "sessions" / "20261109-190507-P10" / k).is_dir()
        assert getattr(d, k) == tmp_path / "sessions" / "20261109-190507-P10" / k
    assert st.SessionDirs(tmp_path, "20261109-190507-P10").exists()


def test_session_id_must_be_a_plain_name(tmp_path):
    for bad in ("../x", "a/b", "", "."):
        with pytest.raises(ValueError):
            st.SessionDirs(tmp_path, bad)


def test_disk_check(tmp_path):
    ok, free = st.disk_check(tmp_path, min_gb=0.0)
    assert ok and free > 0
    ok, free = st.disk_check(tmp_path, min_gb=1e9)
    assert not ok


@pytest.mark.parametrize("p", ["/root/p21-cal/dma_motor_motion", "/root/p21-cal/backup/dma-pre-210-x",
                               "/root/p07-stack/a", "/root/p09-kb4t-backup-20261104/y", "/root/p11-cal/takes/n6_a",
                               "/root/p21-cal/takes"])
def test_protected_defaults(p):
    assert st.is_protected(p)


def test_not_protected_and_extra_refs():
    assert not st.is_protected("/data/calib/sessions/x/raw")
    assert st.is_protected("/data/calib/sessions/x/raw", refs=["/data/calib/sessions/x"])


def S(sid, status, deployed=False):
    return {"id": sid, "status": status, "deployed": deployed}


def test_retention_keeps_live_and_last_two_completed_deploys():
    ss = [S("01", "completed", True), S("02", "aborted"), S("03", "completed", True), S("04", "read-only"),
          S("05", "completed", True), S("06", "completed", True), S("07", "aborted")]
    assert st.retention(ss, live_id="03") == ["01", "02", "04"]


def test_retention_never_deletes_a_running_session():
    ss = [S("01", "running"), S("02", "completed", True), S("03", "completed", True), S("04", "completed", True)]
    assert st.retention(ss, live_id="04") == ["02"]


def test_delete_raw_refuses_protected(tmp_path):
    d = st.SessionDirs(tmp_path, "s1").create()
    (d.raw / "pose_000.npz").write_bytes(b"x")
    with pytest.raises(st.Protected):
        st.delete_raw(d, refs=[str(d.root)])
    assert (d.raw / "pose_000.npz").exists()
    st.delete_raw(d)
    assert not d.raw.exists() and d.corners.is_dir()


def test_list_sessions_reads_their_state(tmp_path):
    import json
    for sid, status in (("a", "aborted"), ("b", "completed")):
        d = st.SessionDirs(tmp_path, sid).create()
        (d.root / "state.json").write_text(json.dumps({"status": status, "deployed": status == "completed"}))
    got = st.list_sessions(tmp_path)
    assert [(s["id"], s["status"], s["deployed"]) for s in got] == [("a", "aborted", False), ("b", "completed", True)]
