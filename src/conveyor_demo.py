"""
conveyor_demo.py -- animated top-down view of the moving-belt simulation (src/conveyor_sim.py).

Three lanes carry the same item (same seed): camera-only, plain rule, and the proposed system. Each lane is
the real conveyor_sim.trial() trace; nothing is scripted. Used by the "Conveyor Line" tab of dashboard.py.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import conveyor_sim as cs

LANES = [("camera_only", "Camera only"), ("rule", "Plain rule"), ("ours", "Proposed")]
LANE_Y = {"camera_only": 2.0, "rule": 1.0, "ours": 0.0}
# The first four are the same items as the Live Monitor tab (same names), shown on a moving belt.
SHARED = {
    "Single battery from room temperature": "intact",
    "Failing cell arriving at 80 C": "fail80",
    "Benign items only": "benign",
    "Gas spikes, no battery": "benign_burst",
}
SCENARIOS = {
    **SHARED,
    "Failing cell hidden in a device (60 C, slow)": "fail60_hidden",
    "Failing cell hidden in a device (80 C)": "fail80_hidden",
    "Warm benign item + gas burst": "warm_burst",
    "Hot ambient 45 C + gas burst": "hot45_burst",
    "Gas sensors dead + hidden cell": "gasdead_fail80_hidden",
    "Open belt (no hood) + hidden cell": "open_fail80_hidden",
}
STATE_COLOR = {"Normal": "#2ecc71", "Warning": "#f1c40f", "Critical": "#e67e22",
               "Emergency_Stop": "#e74c3c", "Sensor_Fault": "#9b59b6"}
X_SHRED_IN = cs.X_SHRED + 0.15
BIN_X = cs.X_DIV


def run(kind, seed, pool):
    """Real trial traces for the three lanes plus the item's own temperature/phase track."""
    lanes = {s: cs.trial(kind, seed, s, pool, trace=True) for s, _ in LANES}
    p = cs.scenario_params(kind)
    n = int(round((cs.T_END + cs.BIN_T) / cs.DT))
    T, ph, h2, _ = cs.item_track(kind, np.random.RandomState(seed), p["amb"], n)
    return lanes, dict(T=T, phase=ph, h2=h2, amb=p["amb"])


def outcome(res, battery):
    fail = res["scenario"].count("fail") > 0
    if res["entered_shredder"]:
        if res["hazardous_at_entry"]: return "SHREDDED: failing cell entered the shredder", "#e74c3c"
        if battery: return "SHREDDED: intact battery crushed", "#e74c3c"
        return "Passed to shredder (correct)", "#2ecc71"
    if res["diverted"]:
        if not battery: return "False divert: one item re-sorted by hand", "#f1c40f"
        if res["system"] == "camera_only" and fail: return "Diverted, but bin unwatched: fire in bin", "#e67e22"
        if fail and res["bin_alarm"]: return f"Diverted ({res['divert_reason']}); bin alarm at {res['t_bin_alarm']:.0f} s", "#2ecc71"
        return f"Diverted to bin ({res['divert_reason']})", "#2ecc71"
    if res["belt_stopped"]:
        return ("BELT STOPPED before shredder" if battery else "FALSE STOP: belt halted"), ("#2ecc71" if battery else "#f1c40f")
    return "Still on belt", "#95a5a6"


def _lane_at(tr, res, t):
    """Item x, lane state and belt speed at time t (after the trace ends the item stays where it finished)."""
    rows = tr[tr.t <= t + 1e-9]
    last = rows.iloc[-1]
    ended = t > tr.t.iloc[-1]
    if ended and res["diverted"]: return BIN_X, "bin", last.state, last.speed
    if ended and res["entered_shredder"]: return X_SHRED_IN, "shredder", last.state, last.speed
    return max(last.x, -0.3), "belt", last.state, last.speed


EV_STYLE = {"camera": "#9b59b6", "hood": "#3498db", "Warning": "#f1c40f", "Critical": "#e67e22",
            "Emergency_Stop": "#e74c3c", "divert": "#f39c12", "bin": "#1abc9c", "shredder": "#c0392b", "runaway": "#ff0000"}


def events(res, tr, item, system):
    """What happened to the item on one lane, with time (s after the camera) and belt position (m)."""
    ev = []
    tr = tr[tr.t >= 0]
    if system == "rule":
        ev.append((0.0, 0.0, "camera", "no camera on this line"))
    else:
        hit = res["P_battery"] >= cs.P_DIVERT
        ev.append((0.0, 0.0, "camera", f"camera P_battery {res['P_battery']:.2f}: " + ("battery seen" if hit else "no battery seen")))
    u = tr[tr.under]
    if len(u):
        ev.append((u.t.iloc[0], u.x.iloc[0], "hood", "enters IR + gas hood"))
    prev = "Normal"
    for r in tr.itertuples():
        if r.state != prev:
            ev.append((r.t, r.x, r.state, f"{r.state.replace('_', ' ')}: belt {r.speed * 100:.0f} %"))
            prev = r.state
    if res["diverted"]:
        ev.append((res["t_divert"], cs.X_DIV, "divert", f"diverter fires ({res['divert_reason']}) -> bin"))
        if res.get("bin_alarm") is True:
            ev.append((res["t_bin_alarm"], cs.X_DIV, "bin", "bin sensors raise the alarm"))
    if res["entered_shredder"]:
        ev.append((res["t_entry"], cs.X_SHRED, "shredder", "enters shredder" + (" while self-heating" if res["hazardous_at_entry"] else "")))
    ru = np.where(np.isin(item["phase"], ["Phase3_Runaway", "Phase4_Fire"]))[0]
    if len(ru):
        ev.append((ru[0] * cs.DT, np.nan, "runaway", "cell itself enters thermal runaway"))
    return sorted(ev, key=lambda e: e[0])


def belt_disp(tr, res, t):
    """How far this lane's belt has moved by time t: it follows the item, runs on after a divert, stays put after a stop."""
    rows = tr[tr.t <= t + 1e-9]
    if not len(rows): return 0.0
    d = rows.x.iloc[-1] - tr.x.iloc[0]
    if t > tr.t.iloc[-1] and not (res["belt_stopped"] or rows.state.iloc[-1] == "Emergency_Stop"):
        d += cs.V_BELT * (t - tr.t.iloc[-1])
    return d


def figure(kind, seed, pool, step=0.25, tail=4.0):
    lanes, item = run(kind, seed, pool)
    battery = any(w in kind for w in ("intact", "fail"))
    t_end = max(min(tr.t.iloc[-1], 60.0) for _, tr in lanes.values()) + tail
    times = np.arange(0.0, t_end + 1e-9, step)

    fig = go.Figure()
    shapes, ann = [], []
    for s, label in LANES:
        y = LANE_Y[s]
        shapes += [dict(type="rect", x0=-0.35, x1=cs.X_SHRED, y0=y - 0.16, y1=y + 0.16, fillcolor="#3d3d3d", line_width=0, layer="below"),
                   dict(type="rect", x0=BIN_X - 0.13, x1=BIN_X + 0.13, y0=y + 0.2, y1=y + 0.44,
                        fillcolor="#5d6d7e" if s != "camera_only" else "#2c3e50", line=dict(color="#aab7b8"), layer="below")]
        ann += [dict(x=-0.42, y=y, text=f"<b>{label}</b>", showarrow=False, xanchor="right", font=dict(size=13)),
                dict(x=BIN_X, y=y + 0.5, text="bin + sensors" if s == "ours" else ("bin" if s == "camera_only" else ""),
                     showarrow=False, font=dict(size=9, color="#aab7b8"))]
        for xr in (-0.35, cs.X_SHRED):
            shapes.append(dict(type="circle", x0=xr - 0.05, x1=xr + 0.05, y0=y - 0.17, y1=y + 0.17, fillcolor="#7f8c8d",
                               line=dict(color="#bdc3c7"), layer="below"))
    shapes += [dict(type="rect", x0=cs.X_HOOD0, x1=cs.X_HOOD1, y0=-0.3, y1=2.3, fillcolor="rgba(52,152,219,0.12)",
                    line=dict(color="#3498db", dash="dot"), layer="below"),
               dict(type="rect", x0=cs.X_SHRED, x1=cs.X_SHRED + 0.3, y0=-0.3, y1=2.3, fillcolor="#1b1b1b",
                    line=dict(color="#e74c3c"), layer="below"),
               dict(type="line", x0=0, x1=0, y0=-0.3, y1=2.3, line=dict(color="#9b59b6", width=2, dash="dash")),
               dict(type="line", x0=BIN_X, x1=BIN_X, y0=-0.3, y1=2.3, line=dict(color="#f39c12", width=2, dash="dash"))]
    for x, txt in ((0, "camera"), ((cs.X_HOOD0 + cs.X_HOOD1) / 2, "IR + gas hood"), (BIN_X, "diverter"), (cs.X_SHRED + 0.15, "SHREDDER")):
        ann.append(dict(x=x, y=2.62, text=txt, showarrow=False, font=dict(size=11)))

    lane_ev = {s: events(lanes[s][0], lanes[s][1], item, s) for s, _ in LANES}
    SLAT = 0.1

    def belt_and_pins(t):
        bx, by, px, py, pt, pc = [], [], [], [], [], []
        for s, _ in LANES:
            res, tr = lanes[s]; y = LANE_Y[s]
            off = belt_disp(tr, res, t) % SLAT
            for xs_ in np.arange(-0.35 + off, cs.X_SHRED, SLAT):
                bx += [xs_, xs_, None]; by += [y - 0.15, y + 0.15, None]
            shown = [e for e in lane_ev[s] if e[0] <= t + 1e-9 and not np.isnan(e[1]) and e[2] not in ("hood",)]
            for k, (te, xe, kind_, _) in enumerate(shown[-4:]):
                px.append(xe); py.append(y - 0.24 - 0.17 * (k % 2)); pc.append(EV_STYLE.get(kind_, "#95a5a6"))
                pt.append(f"{kind_.replace('_', ' ').replace('Emergency Stop', 'E-STOP')} {te:.1f}s")
        return [go.Scatter(x=bx, y=by, mode="lines", line=dict(color="#5a5a5a", width=2), hoverinfo="skip"),
                go.Scatter(x=px, y=py, mode="markers+text", text=pt, textposition="bottom center",
                           textfont=dict(size=10, color=pc), marker=dict(symbol="triangle-up", size=11, color=pc), hoverinfo="text")]

    def frame_data(t):
        xs, ys, cols, txt, gx, gy, gs, sx, sy, st_txt, st_col = [], [], [], [], [], [], [], [], [], [], []
        i = min(int(round(t / cs.DT)), len(item["T"]) - 1)
        for s, _ in LANES:
            res, tr = lanes[s]; y = LANE_Y[s]
            x, where, state, speed = _lane_at(tr, res, t)
            yy = y + 0.32 if where == "bin" else y
            T = item["T"][i] if t >= 0 else item["amb"]
            xs.append(x); ys.append(yy); cols.append(T)
            txt.append(f"{T:.0f} C" + ("" if item["phase"][i] == "Normal" else f"<br>{item['phase'][i].replace('_', ' ')}"))
            plume = item["h2"][i] if x >= 0 else 0.0
            gx.append(x + 0.04 * np.log1p(plume)); gy.append(yy + 0.02 * np.log1p(plume)); gs.append(min(8 + 6 * np.log1p(plume), 70) if plume > 1 else 0)
            sx.append(cs.X_SHRED + 0.5); sy.append(y)
            st_txt.append(f"{state.replace('_', ' ')}<br>belt {speed * 100:.0f} %"); st_col.append(STATE_COLOR.get(state, "#95a5a6"))
        return belt_and_pins(t) + [go.Scatter(x=gx, y=gy, mode="markers", marker=dict(size=gs, color="rgba(170,200,120,0.28)", line_width=0), hoverinfo="skip"),
                go.Scatter(x=xs, y=ys, mode="markers+text", text=txt, textposition="top center", textfont=dict(size=10, color="white"),
                           marker=dict(size=24, symbol="square", color=cols, colorscale="Turbo", cmin=20, cmax=850,
                                       line=dict(color="white", width=1), colorbar=dict(title="cell C", x=1.12, len=0.8))),
                go.Scatter(x=sx, y=sy, mode="markers+text", text=st_txt, textposition="middle right",
                           marker=dict(size=16, color=st_col), textfont=dict(size=11), hoverinfo="skip")]

    for d in frame_data(0.0): fig.add_trace(d)
    frames = []
    for t in times:
        f_ann = ann + [dict(x=1.25, y=-0.62, text=f"t = {t:5.1f} s after the camera", showarrow=False, font=dict(size=12))]
        if t >= times[-1] - tail + step:
            for s, _ in LANES:
                txt, col = outcome(lanes[s][0], battery)
                f_ann.append(dict(x=0.75, y=LANE_Y[s] + 0.36, text=f"<b>{txt}</b>", showarrow=False, font=dict(size=12, color="white"),
                                  bgcolor=col, opacity=0.95))
        frames.append(go.Frame(data=frame_data(t), name=f"{t:.2f}", layout=dict(annotations=f_ann)))
    fig.frames = frames
    fig.update_layout(
        height=560, margin=dict(l=110, r=40, t=30, b=40), showlegend=False, shapes=shapes, annotations=ann,
        xaxis=dict(range=[-0.5, 3.45], showgrid=False, zeroline=False, title="belt position (m)"),
        yaxis=dict(range=[-0.8, 2.8], showgrid=False, zeroline=False, showticklabels=False),
        updatemenus=[dict(type="buttons", x=0.0, y=-0.08, xanchor="left", direction="left", buttons=[
            dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=120, redraw=True), fromcurrent=True, transition=dict(duration=0))]),
            dict(label="Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
        sliders=[dict(x=0.18, y=-0.06, len=0.8, currentvalue=dict(visible=False),
                      steps=[dict(method="animate", label=f"{t:.0f}" if abs(t - round(t)) < 1e-9 and int(round(t)) % 5 == 0 else "",
                                  args=[[f"{t:.2f}"], dict(mode="immediate", frame=dict(duration=0, redraw=True))]) for t in times])])
    fig._lane_events = lane_ev
    return fig, lanes, item


def physics_figure(lanes, item, system="ours"):
    """Sensor readings the chosen lane's controller saw, with its FSM state as a coloured strip."""
    res, tr = lanes[system]; tr = tr[tr.t >= -5]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.45, 0.4, 0.15], vertical_spacing=0.04,
                        specs=[[{"secondary_y": False}], [{"secondary_y": True}], [{"secondary_y": False}]])
    fig.add_trace(go.Scatter(x=tr.t, y=tr.T_item, name="cell temperature (true)", line=dict(color="#e74c3c")), 1, 1)
    fig.add_trace(go.Scatter(x=tr.t, y=tr.T_ir, name="IR reading (hood)", line=dict(color="#f39c12", dash="dot")), 1, 1)
    fig.add_trace(go.Scatter(x=tr.t, y=tr.h2, name="MQ-2 H2 reading (ppm)", line=dict(color="#27ae60")), 2, 1)
    fig.add_trace(go.Scatter(x=tr.t, y=tr.co2, name="MQ-135 CO2 reading (ppm)", line=dict(color="#16a085", dash="dot")), 2, 1, secondary_y=True)
    fig.add_trace(go.Bar(x=tr.t, y=np.ones(len(tr)), marker_color=[STATE_COLOR.get(s, "#95a5a6") for s in tr.state],
                         marker_line_width=0, name="FSM state", hovertext=tr.state, width=0.1), 3, 1)
    for x0, x1 in _spans(tr.t.values, tr.under.values):
        fig.add_vrect(x0=x0, x1=x1, fillcolor="rgba(52,152,219,0.12)", line_width=0, row="all", col=1)
    fig.update_yaxes(title_text="C", row=1, col=1); fig.update_yaxes(title_text="H2 ppm", row=2, col=1)
    fig.update_yaxes(title_text="CO2 ppm", row=2, col=1, secondary_y=True); fig.update_yaxes(showticklabels=False, row=3, col=1)
    fig.update_xaxes(title_text="s after the camera (blue band: item under the hood)", row=3, col=1)
    fig.update_layout(height=430, margin=dict(l=60, r=60, t=20, b=40), legend=dict(orientation="h", y=1.08), bargap=0)
    return fig


def _spans(t, mask):
    out, start = [], None
    for ti, m in zip(t, mask):
        if m and start is None: start = ti
        if not m and start is not None: out.append((start, ti)); start = None
    if start is not None: out.append((start, t[-1]))
    return out


if __name__ == "__main__":          # self-check: every scenario builds and the outcomes are the simulation's
    pool = {"test_battery_coarse": np.array([0.9]), "test_battery_recybat": np.array([0.9]),
            "test_nonbattery": np.array([0.05]), "test_electronics": np.array([0.02])}
    for label, kind in SCENARIOS.items():
        fig, lanes, item = figure(kind, 100, pool)
        assert len(fig.frames) > 10
        for sname, _ in LANES:
            ev = events(lanes[sname][0], lanes[sname][1], item, sname)
            res = lanes[sname][0]
            assert ev[0][2] == "camera"
            assert res["diverted"] == any(e[2] == "divert" for e in ev)
            assert res["entered_shredder"] == any(e[2] == "shredder" for e in ev)
        print(f"{label:<45}", " | ".join(outcome(lanes[s][0], any(w in kind for w in ('intact', 'fail')))[0] for s, _ in LANES))
