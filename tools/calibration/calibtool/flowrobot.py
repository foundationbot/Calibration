"""The full session's robot side: flow.Robot built from calibd's pieces (M5 wiring). Not run against a robot yet: the
M5 live run with the user is its acceptance (design 13, M5 row).

build() picks, per part, the robot backend or, with dry_run, the recording one, the reads staying real:
  motors     motors_enable.Enabler(motors.run_cmd, EnableProbe) -- dry: Enabler(dry_run=True) (records enable/disable)
  neck       neck.Neck(the fixed dma_motor_motion, motors.run_cmd, ShmRobot) -- dry: neck.DryRun started at the
             measured joint angles (the exact argv recorded, nothing sent)
  producer   producer.Kube(kubectl, read_only=False), every change logged -- dry: sessionctl.DryKubectl seeded with the
             robot's producer spec (read from the robot)
  OKVIS, the deploy, robot.env: deploy.Remote over deploy.shell_runner(), dry_run passed through (FullFlow builds them)
  capture    Capture.grab, DetectPool.submit_pair, the loop's ObsBox for the guidance
holds_session() turns the captured holds into the Session the split and the fit read (FIT/TEST labels are keyed by
left stamp, so the take/hold renumbering of Session.from_files does not matter). pipeline_fit_decide() is the fit +
verdict + export (step 45 / 66 through replay.evaluate as they are, step 71 for the OKVIS file); with a live_fit
(refit.LiveFit) its refit on the final split stands for step 45 when it covers exactly the split (refit.final_fit)."""
from pathlib import Path

import numpy as np

from calibtool import deploy as dp, flow, motors, motors_enable as me, neck as nk, producer as pr, sessionctl as sc


def build(prof, *, session, dirs, capture, detect_pair, producer_yaml, probes, baseline, obs_box, guide_kw,
          make_collection, fit_decide, dry_run, run=None, motor_run=None, kube_runner=None, shm_robot=None,
          tool_info=None, imu=None, sweep_detect=None, R_live=None, check_scores=None, post_scores=None,
          okvis_probe=None, plan_next=None, predict=None, imu_fit=None, live_fit=None) -> flow.Robot:
    run = run or dp.shell_runner()
    motor_run = motor_run or motors.run_cmd
    kube_runner = kube_runner or pr.kubectl_runner()
    rec = sc.Recorder(dry_run=dry_run)
    enabler = me.Enabler(motor_run, me.EnableProbe(motor_run, probes["autonomy"], probes["pods"]), baseline=baseline,
                         dry_run=dry_run, adopt_on=True)
    if dry_run:
        live = pr.Kube(kube_runner, prof.producer_ns, prof.producer_deploy, read_only=True).get_yaml()
        kube = pr.Kube(sc.DryKubectl(live, rec), prof.producer_ns, prof.producer_deploy, read_only=False)
    else:
        def logged(argv, input=None):
            if argv and argv[0] not in pr.READ_VERBS:
                rec.add("producer", ["k0s", "kubectl", *argv], input)
            return kube_runner(argv, input=input) if input is not None else kube_runner(argv)
        kube = pr.Kube(logged, prof.producer_ns, prof.producer_deploy, read_only=False)
    shm_robot = shm_robot or nk.ShmRobot("/dev/shm")
    tool = prof.motor_tool

    def make_neck():
        if dry_run:
            names, q = shm_robot.joints()
            dr = nk.DryRun(names, q, tool_path=tool)
            return nk.Neck(tool, dr.run, dr.robot, dr.tool_info, clock=dr.clock, sleep=dr.clock.sleep)
        return nk.Neck(tool, motor_run, shm_robot, tool_info or motors.motor_tool(tool))

    return flow.Robot(
        run=run, dry_run=dry_run, recorder=rec, enabler=enabler,
        motors_on=lambda: motors.master(motor_run).motors_on, make_neck=make_neck, kube=kube,
        lease=pr.Lease(Path(prof.root) / "producer.lease"), producer_yaml=producer_yaml, grab=capture.grab,
        detect=detect_pair, board_obs=obs_box.take, make_collection=make_collection, fit_decide=fit_decide,
        check_scores=check_scores, post_scores=post_scores, okvis_pods=lambda: dp.okvis_pods(dp.Remote(run)),
        deploy_dir=str(dirs.deploy), session=session, guide_kw=dict(guide_kw), predict=predict, plan_next=plan_next,
        imu_accel=None if imu is None else imu.accel, frames0=lambda k: [p.left.y for p in capture.grab(k)],
        sweep_detect=sweep_detect, R_live=R_live, imu_fit=imu_fit or imu_fit_default, okvis_probe=okvis_probe,
        live_fit=live_fit, board_miss=getattr(obs_box, "last_miss", None))


def imu_fit_default(results, R_live) -> dict:
    """imusweep.to_sweep + imu.verdict -> what the flow keeps"""
    from calibtool import imu, imusweep
    v = imu.verdict(imusweep.to_sweep(results, R_live), R_live)
    return {"deploy": v.deploy, "reasons": list(v.reasons), "correction": np.asarray(v.correction).tolist(),
            "change_vs_live_deg": float(v.change_vs_live_deg)}


def holds_session(holds, name="calibtool session"):
    """the captured holds (assign.CaptureHold) as a Session: one take per placement, in the order first captured"""
    from calibtool import postcheck
    from calibtool.session import Session
    order, by = [], {}
    for h in sorted(holds, key=lambda x: (int(x.t0), int(x.hold))):
        if h.placement not in by:
            order.append(h.placement)
            by[h.placement] = []
        by[h.placement].append(h)
    return Session.from_files([(p, postcheck.merge_holds(by[p])) for p in order], name=name)


def pipeline_fit_decide(rig_path, live_config, out_name, gates=None, jobs=1, live_fit=None):
    """flow.Robot.fit_decide on the robot: the split of the capture-time labels, replay.evaluate (step 45, step 66 of
    new and live, decide.score, decide.verdict with the default required edges) and, on DEPLOY, step 71's KB4T OKVIS
    file over the live config as template. live_config: () -> the live OKVIS yaml path. live_fit: refit.LiveFit (the
    flow's): its refit on the final split stands for step 45 when it covers exactly this split's FIT corners and
    passed step 45's checks (refit.final_fit); otherwise step 45 runs. out["fit_source"] says which"""
    def fit_decide(col, holds, out_dir):
        from calibtool import decide, paths, pipeline, replay
        out_dir = Path(out_dir)
        sess = holds_session(holds)
        split = col.labeler.write_split(sess, out_dir / "split")
        live = live_config()
        fit, why = (None, "no live refit") if live_fit is None else live_fit.final(split.fit_npz, out_dir / "fit")
        ev = replay.evaluate(sess, rig_path, split, live, out_dir, gates or decide.Gates(), jobs=jobs, fit=fit)
        v = ev["verdicts"][next(iter(replay.VERDICT_KEYS))]
        src = (f"live refit {fit.entry['source']['live_refit']} (step 45 not run)" if fit is not None
               else f"step 45 ({why})")
        out = {"decision": v["decision"], "reasons": list(v["reasons"]), "name": out_name, "config_text": None,
               "fit_source": src}
        if v["decision"] == "DEPLOY":
            ex = out_dir / "export"
            rc, _ = pipeline._run(paths.script("71_export_lens"),
                                  [ev["fit"]["json_path"], ev["validate_new"]["report_path"], ex, "--okvis-template",
                                   live, "--okvis-kb4t", "--rig", rig_path], out_dir / "71.log")
            f = ex / pipeline.load_rig(rig_path).okvis_yaml_name
            if rc != 0 or not f.is_file():
                return {**out, "decision": "NOT_ENOUGH_DATA",
                        "reasons": out["reasons"] + [f"step 71 wrote no OKVIS file (exit {rc}): see {out_dir / '71.log'}"]}
            out["config_text"] = f.read_text()
        return out
    return fit_decide


def pairs_to_holds(pairs, detect_pair, hold=0, placement="postcheck", min_tags=None):
    """captured pairs -> one assign.CaptureHold of the pairs with the board in both cameras, and the median left/right
    stamp gap (ms)"""
    from calibtool import assign, detect, protocol
    min_tags = protocol.MIN_TAGS if min_tags is None else min_tags
    rows = []
    for p in pairs:
        r = detect_pair(p.left.y, p.right.y)
        dl, dr = r.result() if hasattr(r, "result") else r
        if len({t for t, _ in dl}) >= min_tags and len({t for t, _ in dr}) >= min_tags:
            rows.append((int(p.t0), int(p.t1), dl, dr))
    sync = float(np.median([abs(p.dt_ns) for p in pairs])) / 1e6 if pairs else None
    if not rows:
        return [], sync
    return [assign.CaptureHold(hold, placement, rows[0][0], detect.hold_dict(rows, hold))], sync


def scorers(rig_path, live_text, work_dir, detect_pair):
    """(check_scores, post_scores) for flow.Robot: a config scored on held frames by decide.score's method
    (postcheck.score_holds), the new one and the live one (live_text() -> its yaml) on the same frames, at the frames'
    size"""
    from calibtool import postcheck
    work_dir = Path(work_dir)

    def score(text, holds, size, name):
        work_dir.mkdir(parents=True, exist_ok=True)
        y = work_dir / f"{name}.yaml"
        y.write_text(text)
        js = postcheck.calib_json_for(y, size, work_dir, name)
        return postcheck.score_holds(js, holds, rig_path, work_dir, placement=name)

    def check_scores(holds, size, text):
        return score(text, holds, size, "check-new"), score(live_text(), holds, size, "check-live")

    def post_scores(pairs, text, size):
        holds, sync = pairs_to_holds(pairs, detect_pair)
        if not holds:
            raise ValueError("no post-check pair with the board in both cameras")
        return sync, score(text, holds, size, "post-new"), score(live_text(), holds, size, "post-old")
    return check_scores, post_scores
