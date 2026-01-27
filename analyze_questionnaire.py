# analyze_questionnaire.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import rcParams

# =============================
# CONFIG
# =============================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULT_DIR = BASE_DIR / "results"
PLOT_DIR = RESULT_DIR / "plots"

CSV_PATH = DATA_DIR / "responses.csv"

LETTERS = ["A", "B", "C", "D", "E"]
FEATURES = ["tempo", "rhythm", "vocal", "melody", "genre_sim"]

# 表示順（内部では未回答を保持）
MUSIC_HOURS_ORDER = ["未回答", "全く聴かない", "３０分未満", "１時間", "２時間以上"]
AGE_ORDER = ["未回答", "１０代", "２０代", "３０代", "４０代以上"]
GENDER_ORDER = ["未回答", "男性", "女性", "その他", "回答しない"]

# プロット用（未回答を除外）
MUSIC_HOURS_PLOT_ORDER = ["全く聴かない", "３０分未満", "１時間", "２時間以上"]
AGE_PLOT_ORDER = ["１０代", "２０代", "３０代", "４０代以上"]
GENDER_PLOT_ORDER = ["男性", "女性", "その他", "回答しない"]

RESULT_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

# 日本語フォント
rcParams["font.family"] = ["Meiryo", "MS Gothic", "Yu Gothic", "Noto Sans CJK JP", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

# =============================
# Helpers
# =============================
def safe_to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def ci95(series: pd.Series) -> tuple[float, float]:
    x = series.dropna().astype(float)
    n = len(x)
    if n <= 1:
        return (np.nan, np.nan)
    m = x.mean()
    sd = x.std(ddof=1)
    lo, hi = stats.t.interval(0.95, df=n-1, loc=m, scale=sd / np.sqrt(n))
    return float(lo), float(hi)

def savefig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()

def plot_bar_counts(series: pd.Series, order: list[str], title: str, outpath: Path):
    vc = series.value_counts()
    labels = [x for x in order if x in vc.index]
    values = [int(vc.get(x, 0)) for x in labels]

    plt.figure(figsize=(7, 3.5))
    plt.bar(labels, values)
    plt.title(title)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("count")
    savefig(outpath)

# =============================
# Load
# =============================

if not CSV_PATH.exists():
    raise FileNotFoundError(f"responses.csv not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

# 重複除去
if "timestamp" in df.columns and "session_id" in df.columns:
    df["_ts"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.sort_values("_ts")
    df = df.drop_duplicates(subset=["session_id"], keep="last")
    df = df.drop(columns=["_ts"], errors="ignore")

print("Loaded rows:", len(df))

# 欠損 → 未回答
for col in ["music_hours_per_day", "age", "gender"]:
    if col in df.columns:
        df[col] = df[col].fillna("未回答").astype(str)

# カテゴリ順固定
if "music_hours_per_day" in df.columns:
    df["music_hours_per_day"] = pd.Categorical(df["music_hours_per_day"], MUSIC_HOURS_ORDER, ordered=True)
if "age" in df.columns:
    df["age"] = pd.Categorical(df["age"], AGE_ORDER, ordered=True)
if "gender" in df.columns:
    df["gender"] = pd.Categorical(df["gender"], GENDER_ORDER, ordered=True)

# =============================
# 1) 度数分布（未回答除外）
# =============================
if "music_hours_per_day" in df.columns:
    df_plot = df[df["music_hours_per_day"] != "未回答"]
    plot_bar_counts(df_plot["music_hours_per_day"], MUSIC_HOURS_PLOT_ORDER,
                    "Music listening per day", PLOT_DIR / "dist_music_hours.png")

if "age" in df.columns:
    df_plot = df[df["age"] != "未回答"]
    plot_bar_counts(df_plot["age"], AGE_PLOT_ORDER,
                    "Age group", PLOT_DIR / "dist_age.png")

if "gender" in df.columns:
    df_plot = df[df["gender"] != "未回答"]
    plot_bar_counts(df_plot["gender"], GENDER_PLOT_ORDER,
                    "Gender", PLOT_DIR / "dist_gender.png")

# =============================
# 2) クロス集計（未回答除外）
# =============================
if "music_hours_per_day" in df.columns and "age" in df.columns:
    df_ct = df[
        (df["music_hours_per_day"] != "未回答") &
        (df["age"] != "未回答")
    ]

    ct = pd.crosstab(
        df_ct["music_hours_per_day"],
        df_ct["age"]
    ).reindex(
        index=MUSIC_HOURS_PLOT_ORDER,
        columns=AGE_PLOT_ORDER,
        fill_value=0
    )

    ct.plot(kind="bar", stacked=True, figsize=(8, 4))
    plt.title("Crosstab (count): music_hours_per_day x age")
    plt.ylabel("count")
    plt.xticks(rotation=20, ha="right")
    savefig(PLOT_DIR / "crosstab_music_hours_x_age.png")

print("\n=== DONE ===")
print("Saved plots to:", PLOT_DIR)
