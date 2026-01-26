# app.py
# -*- coding: utf-8 -*-
"""
Streamlit app for Jamendo-based Music Recommender (Experiment Version) - Drive Playback + pop reserve.

改善反映:
- 起動時のサイドバーは閉じた状態にする
- 被験者のバイアスを避けるため、画面上のジャンル表示は出さない（管理者用にのみ表示）
- Top-Kは固定(5)にし、被験者が触れないようにする
- ガイド文言を「評価」→「聴いてください」へ変更（順位提示しない前提）
- Top5表（DataFrame）表示を削除
- 推薦地図(PCA)を小さく＋ズームアウト表示
- ★修正：Top5ボタンは“いつでも押せる”（待機時間ゲート撤廃）

前提:
- index/index_emb64_l2.npy
- index/index_emb64_l2_meta.csv（drive_file_id 列を含む）
"""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ダウンロード用（requestsが無い環境でも動くようにフォールバック）
try:
    import requests  # type: ignore
    _HAS_REQUESTS = True
except Exception:
    _HAS_REQUESTS = False
    import urllib.request


###############################################################################
# CONFIG
###############################################################################
BASE_DIR = Path(__file__).resolve().parent

INDEX_DIR = BASE_DIR / "index"
EMB_NPY_NAME = "index_emb64_l2.npy"
EMB_META_NAME = "index_emb64_l2_meta.csv"

TOPK_FIXED = 5  # ★固定（変更不可）
RANDOM_SEED = None

GOOGLE_FORM_URL = "https://docs.google.com/forms/d/e/1FAIpQLScGFzdmKsTP-nuGLWD_Awh7IHT7utFd5VCuu1Dc54PNTQY0Kw/viewform"

# 割当ジャンル（popは予備枠）
PRIMARY_GENRES: List[str] = ["classical", "jazz", "rock", "hiphop"]
RESERVE_GENRE: str = "pop"
PRIMARY_ORDER: List[str] = ["classical", "jazz", "rock", "hiphop"]  # 順番指定
PRIMARY_LIMIT: int = 10
POP_LIMIT: int = 10  # pop予備枠

# クエリ曲（track_id固定）
QUERY_TRACKS: Dict[str, str] = {
    "classical": "1077954",
    "jazz": "1069786",
    "rock": "1134644",
    "hiphop": "1157595",
    "pop": "1030923",
}

# 永続ファイル（Streamlit Cloudは再デプロイで消える可能性あり）
COUNTERS_FILE = BASE_DIR / "genre_counter.json"
LOG_FILE = BASE_DIR / "sessions_log.csv"
COMPLETED_FILE = BASE_DIR / "completed_sessions.json"

# Google Form entry IDs（あなたのフォームに合わせて置換）
FORM_ENTRY_IDS: Dict[str, str] = {
    "session_id": "entry.1234567890",
    "assigned_genre": "entry.1234567891",
    "query_track_id": "entry.1234567892",
    "rec_track_id_1": "entry.1234567893",
    "rec_track_id_2": "entry.1234567894",
    "rec_track_id_3": "entry.1234567895",
    "rec_track_id_4": "entry.1234567896",
    "rec_track_id_5": "entry.1234567897",
    "true_rank_1": "entry.1234567898",
    "true_rank_2": "entry.1234567899",
    "true_rank_3": "entry.1234567900",
    "true_rank_4": "entry.1234567901",
    "true_rank_5": "entry.1234567902",
    "sim_1": "entry.1234567903",
    "sim_2": "entry.1234567904",
    "sim_3": "entry.1234567905",
    "sim_4": "entry.1234567906",
    "sim_5": "entry.1234567907",
    "shuffle_pos_1": "entry.1234567908",
    "shuffle_pos_2": "entry.1234567909",
    "shuffle_pos_3": "entry.1234567910",
    "shuffle_pos_4": "entry.1234567911",
    "shuffle_pos_5": "entry.1234567912",
}


###############################################################################
# Utils: math / PCA
###############################################################################
def l2_unit(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = float(np.linalg.norm(x))
    return x / (n + eps)

def cosine_topk(
    V: np.ndarray,
    q: np.ndarray,
    topk: int,
    exclude_idx: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    sims = V @ q
    if exclude_idx is not None and 0 <= exclude_idx < len(sims):
        sims[exclude_idx] = -1e9
    idx = np.argsort(-sims)[:topk]
    return idx, sims[idx]

def pca2d_svd(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T

def normalize_track_id(x: object) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    return Path(s).stem


###############################################################################
# Utils: persistence (counters, completed)
###############################################################################
def _load_json(path: Path, default):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default

def _save_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{int(time.time())}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

def load_counters() -> Dict[str, int]:
    data = _load_json(COUNTERS_FILE, default={})
    for g in PRIMARY_GENRES + [RESERVE_GENRE]:
        data[g] = int(data.get(g, 0))
    return data

def save_counters(counters: Dict[str, int]) -> None:
    _save_json_atomic(COUNTERS_FILE, counters)

def load_completed() -> set[str]:
    data = _load_json(COMPLETED_FILE, default={"completed": []})
    return set(data.get("completed", []))

def save_completed(completed: set[str]) -> None:
    _save_json_atomic(COMPLETED_FILE, {"completed": sorted(list(completed))})

def new_session_id() -> str:
    return uuid.uuid4().hex

def assign_genre_pop_reserve(counters: Dict[str, int]) -> Optional[str]:
    for g in PRIMARY_ORDER:
        if counters.get(g, 0) < PRIMARY_LIMIT:
            return g
    if counters.get(RESERVE_GENRE, 0) < POP_LIMIT:
        return RESERVE_GENRE
    return None

def log_session_csv(session_id: str, genre: str, ts: float, form_url: str) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{
        "timestamp": int(ts),
        "session_id": session_id,
        "assigned_genre": genre,
        "form_url": form_url,
    }])
    if LOG_FILE.exists():
        row.to_csv(LOG_FILE, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        row.to_csv(LOG_FILE, mode="w", header=True, index=False, encoding="utf-8-sig")


###############################################################################
# Utils: Google Form prefill
###############################################################################
def build_prefilled_form_url(base_url: str, entry_map: Dict[str, str]) -> str:
    from urllib.parse import urlencode
    params = {k: v for k, v in entry_map.items() if k}
    qs = urlencode(params)
    sep = "&" if "?" in base_url else "?"
    return base_url + sep + qs


###############################################################################
# Utils: Google Drive download
###############################################################################
_CONFIRM_RE = re.compile(r"confirm=([0-9A-Za-z_]+)")

def drive_download_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}"

def _download_bytes_via_requests(url: str, timeout: int = 30) -> bytes:
    sess = requests.Session()
    r = sess.get(url, stream=True, timeout=timeout)
    r.raise_for_status()

    ctype = (r.headers.get("content-type") or "").lower()
    if "text/html" in ctype:
        html = r.text
        m = _CONFIRM_RE.search(html)
        if m:
            confirm = m.group(1)
            url2 = url + f"&confirm={confirm}"
            r2 = sess.get(url2, stream=True, timeout=timeout)
            r2.raise_for_status()
            return r2.content

    return r.content

def _download_bytes_via_urllib(url: str, timeout: int = 30) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()

@st.cache_data(show_spinner=False, ttl=60 * 60, max_entries=500)
def download_mp3_bytes_from_drive(file_id: str) -> bytes:
    if not file_id:
        raise ValueError("drive_file_id is empty")
    url = drive_download_url(file_id)
    if _HAS_REQUESTS:
        return _download_bytes_via_requests(url)
    return _download_bytes_via_urllib(url)


###############################################################################
# Load assets
###############################################################################
@st.cache_resource(show_spinner=False)
def load_assets() -> Tuple[np.ndarray, pd.DataFrame, Dict[str, int]]:
    npy_path = INDEX_DIR / EMB_NPY_NAME
    meta_path = INDEX_DIR / EMB_META_NAME

    if not npy_path.exists():
        raise FileNotFoundError(f"Embedding index not found: {npy_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Embedding meta not found: {meta_path}")

    V = np.load(npy_path).astype(np.float32)
    meta = pd.read_csv(meta_path)

    n = min(len(V), len(meta))
    V = V[:n]
    meta = meta.iloc[:n].reset_index(drop=True)

    if "drive_file_id" not in meta.columns:
        raise ValueError("meta に drive_file_id 列がありません。drive_file_id付きmetaに差し替えてください。")

    V = np.stack([l2_unit(v) for v in V], axis=0)

    info = {
        "index_rows": int(len(V)),
        "dim": int(V.shape[1]) if len(V) > 0 else 0,
        "has_requests": int(_HAS_REQUESTS),
    }
    return V, meta, info


###############################################################################
# UI
###############################################################################
st.set_page_config(
    page_title="Music Recommender (Experiment)",
    layout="wide",
    initial_sidebar_state="collapsed",  # ★サイドバーを閉じた状態で起動
)

st.title("🎧 Music Recommender – Experiment (Drive Playback / pop reserve)")

# CSS for guide
st.markdown(
    """
<style>
@keyframes blink { 0%{opacity:1;} 50%{opacity:0.25;} 100%{opacity:1;} }
.guide {
  padding: 10px 12px;
  border-radius: 12px;
  background: #f1f7ff;
  border: 1px solid rgba(0,0,0,0.08);
  margin: 4px 0 8px 0;
}
.blink { animation: blink 1.0s infinite; }
</style>
""",
    unsafe_allow_html=True,
)

def guide(text: str, active: bool) -> None:
    cls = "guide blink" if active else "guide"
    st.markdown(f"<div class='{cls}'>{text}</div>", unsafe_allow_html=True)

# Load
try:
    V, meta, info = load_assets()
except Exception as e:
    st.error(f"読み込みに失敗しました: {e}")
    st.stop()

if len(V) == 0:
    st.error("インデックスが空です。")
    st.stop()

# Session init
if "initialised" not in st.session_state:
    st.session_state["initialised"] = True

    counters = load_counters()
    completed = load_completed()

    assigned = assign_genre_pop_reserve(counters)
    if assigned is None:
        st.session_state["closed"] = True
        st.session_state["assigned_genre"] = None
        st.session_state["session_id"] = None
    else:
        st.session_state["closed"] = False
        st.session_state["assigned_genre"] = assigned
        st.session_state["session_id"] = new_session_id()
        st.session_state["completed"] = False

        st.session_state["query_track_id"] = QUERY_TRACKS.get(assigned, "")

        # base_idx を track_id で検索
        base_idx: Optional[int] = None
        qid = normalize_track_id(st.session_state["query_track_id"])
        if qid and "track_id" in meta.columns:
            mask = meta["track_id"].astype(str).map(normalize_track_id) == qid
            if mask.any():
                base_idx = int(np.argmax(mask.values))

        if base_idx is None:
            # 見つからない場合はランダム（実験では避けたい）
            rng_tmp = np.random.default_rng(RANDOM_SEED)
            base_idx = int(rng_tmp.integers(0, len(V)))
            st.session_state["query_track_id"] = str(meta.loc[base_idx].get("track_id", ""))

        st.session_state["base_idx"] = base_idx

        # phase: 1=案内 / 3=結果表示 / 4=フォーム誘導
        st.session_state["phase"] = 1
        st.session_state["step3_time"] = None

        st.session_state["rng"] = np.random.default_rng(RANDOM_SEED)

        st.session_state["topk_idx"] = None
        st.session_state["topk_sim"] = None
        st.session_state["shuffle_order"] = None
        st.session_state["prefilled_url"] = None

# Closed
if st.session_state.get("closed"):
    st.error("全ジャンルが満員です（実験終了）。")
    st.stop()

# Sidebar（管理者用：expanderで閉じた状態）
with st.sidebar:
    with st.expander("管理者情報（クリックで展開）", expanded=False):
        st.write(f"Index rows: {info['index_rows']}")
        st.write(f"Embedding dim: {info['dim']}")
        st.write(f"requests available: {bool(info['has_requests'])}")
        st.write("---")
        st.write(f"Assigned genre: {st.session_state.get('assigned_genre')}")
        st.write(f"Session ID: {st.session_state.get('session_id')}")
        st.write(f"Query track_id: {st.session_state.get('query_track_id')}")

# Phase update (ゲート撤廃：ボタンは常に押せるが、フォーム誘導はTop5後に進める)
now = time.time()
if st.session_state.get("phase") == 3 and st.session_state.get("step3_time") is not None:
    if now - st.session_state["step3_time"] >= 20:
        st.session_state["phase"] = 4


###############################################################################
# Playback helper
###############################################################################
def play_by_meta_row(row: pd.Series, label: str = "") -> None:
    file_id = str(row.get("drive_file_id", "") or "").strip()
    if not file_id:
        st.warning("⚠️ 音源が再生できません（drive_file_id が空です）。")
        return
    try:
        audio_bytes = download_mp3_bytes_from_drive(file_id)
        if label:
            st.caption(label)
        st.audio(audio_bytes, format="audio/mp3")
    except Exception as e:
        st.warning("⚠️ Drive から音源取得に失敗しました。")
        st.caption(f"debug: drive_file_id={file_id} / error={e}")


###############################################################################
# Step1: Query
###############################################################################
guide("① まず、基準となる曲を聴いてください（30秒ほど推奨）。", st.session_state.get("phase") in (1,))

base_idx = int(st.session_state["base_idx"])
base_row = meta.iloc[base_idx]

st.subheader("🎵 基準曲 (Query Track)")
title = str(base_row.get("title", "") or "")
artist = str(base_row.get("artist", "") or "")
if title or artist:
    st.markdown(f"**{title}**  —  {artist}".strip())
else:
    st.markdown(f"**track_id={base_row.get('track_id', '')}**")

play_by_meta_row(base_row)

###############################################################################
# Step2: Top5 (いつでも押せる)
###############################################################################
guide("② いつでも押して Top5 を表示できます。", True)

run = st.button("🔎 この曲からTop5を表示", type="primary")

if run:
    q_vec = V[base_idx]
    idx_arr, sim_arr = cosine_topk(V, q_vec, topk=TOPK_FIXED, exclude_idx=base_idx)

    st.session_state["topk_idx"] = idx_arr
    st.session_state["topk_sim"] = sim_arr
    st.session_state["phase"] = 3
    st.session_state["step3_time"] = time.time()

    rng = st.session_state["rng"]
    st.session_state["shuffle_order"] = rng.permutation(len(idx_arr))

    st.rerun()

###############################################################################
# Step3: Show results (shuffled) + PCA map
###############################################################################
if st.session_state.get("topk_idx") is not None:
    idx_arr = st.session_state["topk_idx"]
    sim_arr = st.session_state["topk_sim"]
    order = st.session_state.get("shuffle_order")

    rows_true: List[dict] = []
    for rank, (i_val, s_val) in enumerate(zip(idx_arr, sim_arr), start=1):
        i_int = int(i_val)
        r = meta.iloc[i_int]
        rows_true.append({
            "true_rank": rank,
            "similarity": float(np.round(s_val, 6)),
            "index": i_int,
            "track_id": str(r.get("track_id", "")),
            "title": str(r.get("title", "") or ""),
            "artist": str(r.get("artist", "") or ""),
        })

    if order is not None:
        rows_disp = [rows_true[i] for i in order]
    else:
        rows_disp = rows_true

    letters = ["A", "B", "C", "D", "E"]

    guide("③ 表示された5曲を聴いてください（表示順はランキング順ではありません）。", st.session_state.get("phase") == 3)

    st.markdown("### 🗺️ 推薦地図（PCA 2D）")
    st.caption("★ が基準曲です。数字は（ランキング順の）1〜5です。")
    try:
        X_map = np.vstack([V[base_idx][None, :], V[idx_arr]])
        Z_map = pca2d_svd(X_map)
        Z_map = Z_map - Z_map[0]

        fig, ax = plt.subplots(figsize=(3.2, 2.2), dpi=140)
        ax.scatter(Z_map[1:, 0], Z_map[1:, 1], s=16)
        ax.scatter([0], [0], s=70, marker="*")
        for rnk, (xv, yv) in enumerate(Z_map[1:], start=1):
            ax.text(xv, yv, str(rnk), fontsize=8)

        lim = float(np.max(np.abs(Z_map))) if Z_map.size else 1.0
        lim = max(lim * 2.2, 0.5)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

        ax.set_title("Recommendation Map (PCA 2D)", fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=7)
        st.pyplot(fig, clear_figure=True)
    except Exception as e:
        st.warning(f"PCA 地図の生成に失敗しました: {e}")

    st.markdown("### 🎧 推薦曲プレビュー（A〜E）")
    for letter, r in zip(letters, rows_disp):
        header = f"{letter}"
        if r["title"]:
            header += f": {r['title']}"
        if r["artist"]:
            header += f" — {r['artist']}"
        st.markdown(f"**{header}**")
        play_by_meta_row(meta.iloc[int(r["index"])])

    # Prefilled URL（1回だけ）
    if st.session_state.get("prefilled_url") is None:
        params: Dict[str, str] = {}
        sid = st.session_state.get("session_id", "") or ""
        assigned_genre = st.session_state.get("assigned_genre", "") or ""

        params[FORM_ENTRY_IDS.get("session_id", "")] = sid
        params[FORM_ENTRY_IDS.get("assigned_genre", "")] = assigned_genre
        params[FORM_ENTRY_IDS.get("query_track_id", "")] = str(st.session_state.get("query_track_id", "") or "")

        for pos, r in enumerate(rows_true, start=1):
            params[FORM_ENTRY_IDS.get(f"rec_track_id_{pos}", "")] = str(r["track_id"])
            params[FORM_ENTRY_IDS.get(f"true_rank_{pos}", "")] = str(r["true_rank"])
            params[FORM_ENTRY_IDS.get(f"sim_{pos}", "")] = str(r["similarity"])

        shuffle_pos_map: Dict[int, int] = {}
        if order is not None:
            for disp_pos, true_index in enumerate(order.tolist(), start=1):
                shuffle_pos_map[true_index] = disp_pos
        for pos in range(1, TOPK_FIXED + 1):
            params[FORM_ENTRY_IDS.get(f"shuffle_pos_{pos}", "")] = str(shuffle_pos_map.get(pos - 1, pos))

        prefilled_url_val = build_prefilled_form_url(GOOGLE_FORM_URL, params)
        st.session_state["prefilled_url"] = prefilled_url_val

        try:
            log_session_csv(sid, assigned_genre, time.time(), prefilled_url_val)
        except Exception as e:
            st.warning(f"ログ書き込みに失敗しました: {e}")

    # Step4
    guide("④ 最後にアンケート（Googleフォーム）に回答してください。", st.session_state.get("phase") == 4)

    if st.session_state.get("phase") >= 4 and st.session_state.get("prefilled_url"):
        st.link_button("✅ Googleフォームへ進む", st.session_state["prefilled_url"])

        if not st.session_state.get("completed"):
            if st.button("🎉 完了 (クリックして参加を完了)"):
                counters = load_counters()
                completed_ids = load_completed()
                sid = st.session_state.get("session_id", "") or ""
                g = st.session_state.get("assigned_genre", "") or ""

                if sid and sid not in completed_ids:
                    completed_ids.add(sid)
                    counters[g] = int(counters.get(g, 0)) + 1
                    try:
                        save_counters(counters)
                        save_completed(completed_ids)
                    except Exception as e:
                        st.warning(f"保存に失敗しました: {e}")

                st.session_state["completed"] = True
                st.success("ご参加ありがとうございました。回答が記録されました。ブラウザを閉じて構いません。")

        else:
            st.success("このセッションは既に完了しました。ありがとうございました！")

else:
    st.caption("※ まだTop5を表示していません。上のボタンを押してください。")
