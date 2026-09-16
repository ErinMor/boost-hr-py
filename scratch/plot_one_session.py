from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============== USER INPUTS (edit these) ==============
CSV_PATH = Path("/path/to/sub8002_wk1_ses01.csv")
WEEK = 1  # supervised week (1-6)

# If you already know the zone floor for the week, set it directly:
LOWEST_ALLOWED_BPM = 110

# Output image
OUT_PNG = Path("./session_plot.png")


# ============== bounded plan (matches zone_qc.py supervised plan) ==============
SUP_PLAN = {
    1: {"warmup_min": 5, "bounded_min": 15, "unbounded_min": 15, "cooldown_min": 5},
    2: {"warmup_min": 5, "bounded_min": 20, "unbounded_min": 10, "cooldown_min": 5},
    3: {"warmup_min": 5, "bounded_min": 25, "unbounded_min": 5,  "cooldown_min": 5},
    4: {"warmup_min": 5, "bounded_min": 30, "unbounded_min": 0,  "cooldown_min": 5},
    5: {"warmup_min": 5, "bounded_min": 30, "unbounded_min": 0,  "cooldown_min": 5},
    6: {"warmup_min": 5, "bounded_min": 30, "unbounded_min": 0,  "cooldown_min": 5},
}


def load_polar_timeseries(csv_path: Path) -> pd.DataFrame:
    """Load Polar-export CSVs that have a metadata section followed by a time-series table."""
    # Find the time-series header line (where the real table starts)
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    start_idx = None
    for i, line in enumerate(lines):
        if "Sample rate" in line and "Time" in line and "HR" in line:
            start_idx = i
            break
    if start_idx is None:
        raise ValueError("Could not find time-series header (line containing Sample rate/Time/HR).")

    df = pd.read_csv(csv_path, skiprows=start_idx)
    df.columns = [c.strip() for c in df.columns]

    # Standardize time + HR
    if "Time" not in df.columns:
        raise ValueError(f"Missing 'Time' column. Columns found: {list(df.columns)}")

    # Common HR column name in exports is "HR (bpm)"
    hr_col = "HR (bpm)" if "HR (bpm)" in df.columns else "HR"
    if hr_col not in df.columns:
        raise ValueError(f"Missing HR column. Columns found: {list(df.columns)}")

    df["t_sec"] = pd.to_timedelta(df["Time"].astype(str), errors="coerce").dt.total_seconds()
    df["hr"] = pd.to_numeric(df[hr_col], errors="coerce")
    df = df.dropna(subset=["t_sec"]).sort_values("t_sec").reset_index(drop=True)
    return df[["t_sec", "hr"]]


def bounded_window_seconds(week: int) -> tuple[float, float]:
    plan = SUP_PLAN[int(week)]
    warmup_s = plan["warmup_min"] * 60
    bounded_s = plan["bounded_min"] * 60
    start = warmup_s
    end = warmup_s + bounded_s
    return start, end


def dip_metrics_bounded(df: pd.DataFrame, lowest_allowed_bpm: float, week: int) -> dict:
    b0, b1 = bounded_window_seconds(week)
    in_bounded = (df["t_sec"] >= b0) & (df["t_sec"] <= b1)
    d = df.loc[in_bounded].copy()

    # Per-sample duration using next-sample delta (last uses median)
    t = d["t_sec"].to_numpy()
    dt = np.diff(t, append=np.nan)
    med = np.nanmedian(dt)
    if not np.isfinite(med) or med < 0:
        med = 1.0
    dt = np.where(np.isfinite(dt), dt, med)
    dt = np.clip(dt, 0, None)

    below = (d["hr"] < lowest_allowed_bpm).fillna(False).to_numpy().astype(int)
    changes = np.diff(below, prepend=0, append=0)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    # Bout durations in seconds
    bout_durs = []
    for s, e in zip(starts, ends):
        bout_durs.append(float(dt[s:e].sum()))

    dip_count = len(bout_durs)
    dip_total_s = float(sum(bout_durs))
    dip_longest_s = float(max(bout_durs)) if bout_durs else 0.0
    return {
        "dip_count": dip_count,
        "dip_total_time_s": dip_total_s,
        "dip_longest_s": dip_longest_s,
        "bounded_start_s": float(b0),
        "bounded_end_s": float(b1),
    }


def plot_session(df: pd.DataFrame, lowest_allowed_bpm: float, week: int, out_png: Path):
    b0, b1 = bounded_window_seconds(week)
    t_min = df["t_sec"] / 60

    plt.figure(figsize=(12, 4))
    plt.plot(t_min, df["hr"], color="black", linewidth=1)
    plt.axhline(lowest_allowed_bpm, color="red", linestyle="--", label=f"zone floor = {lowest_allowed_bpm:.0f} bpm")

    # Shade bounded window
    plt.axvspan(b0/60, b1/60, color="dodgerblue", alpha=0.12, label="bounded window")

    # Highlight below-floor within bounded
    in_bounded = (df["t_sec"] >= b0) & (df["t_sec"] <= b1)
    below = in_bounded & (df["hr"] < lowest_allowed_bpm)
    plt.scatter(t_min[below], df.loc[below, "hr"], s=8, color="red", alpha=0.6)

    plt.xlabel("Minutes from start")
    plt.ylabel("HR (bpm)")
    plt.title(f"{CSV_PATH.name} (week {week})")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


if __name__ == "__main__":
    df = load_polar_timeseries(CSV_PATH)
    metrics = dip_metrics_bounded(df, LOWEST_ALLOWED_BPM, WEEK)
    print(metrics)
    plot_session(df, LOWEST_ALLOWED_BPM, WEEK, OUT_PNG)
    print(f"Saved: {OUT_PNG.resolve()}")