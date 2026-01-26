# app.py
# -*- coding: utf-8 -*-
"""
Streamlit app for Jamendo-based Music Recommender (Experiment Version) - Drive Playback + pop reserve.

- Cloud公開前提（ローカルaudio不要）
- meta(index_emb64_l2_meta.csv) の drive_file_id を使って Drive から mp3 をDLして再生
- 参加者割当は「4ジャンル(各10人) + pop別枠(予備枠)」に対応
- カウントは「完了ボタン」押下時のみ加算（未完は無視）
- session_id の二重加算防止あり（completed_sessions.json）

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

TOPK_DEFAULT = 5
RANDOM_SEED = None

GOOGLE_FORM_URL = "https://docs.google.com/forms/d/e/1FAIpQLScGFzdmKsTP-nuGLWD_Awh7IHT7utFd5VCuu1Dc54PNTQY0Kw/viewform"

# 割当ジャンル（popは予備枠）
PRIMARY_GENRES: List[str] = ["classical", "jazz", "rock", "hiphop"]
RESERVE_GENRE: str = "pop"

PRIMARY_ORDER: List[str] = ["classical", "jazz", "rock", "hiphop"]  # 順番指定
PRIMARY_LIMIT: int = 10
POP_LIMIT: int = 10  # ← popを別枠で何人まで許すか（例: 20にすれば20人）

# クエリ曲（track_id で固定）
QUERY_TRACKS: Dict[str, str] = {
    "classical": "1077954",
    "jazz": "1069786",
    "rock": "1134644",
    "hiphop": "1157595",
    "pop": "1030923",
}

# 永続ファイル（Cloudでは同一コンテナ内では残るが、再デプロイで消える可能性あり）
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

def cosine_topk(V: np.ndarray, q: np.ndarray, topk: int, exclude_idx: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
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
    # 既定キーを埋める
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
    """
    順番指定：classical→jazz→rock→hiphop の順で各10人まで埋める。
    4ジャンルが全て上限に達したら pop を POP_LIMIT まで割り当てる。
    全て満員なら None。
    """
    for g in PRIMARY_ORDER:
        if counters.get(g, 0) < PRIMARY_LIMIT:
            return g
    if counters.get(RESERVE_GENRE, 0) < POP_LIMIT:
        return RESERVE_GENRE
    return None

def log_session_csv(session_id: str, genre: str, ts: float, form_url: str) -> None:
    """
    sessions_log.csv に追記。失敗しても致命ではないので例外は呼び出し側で握る。
    """
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
    """
    entry_map: {"entry.xxxxx": "value", ...}
    """
    from urllib.parse import urlencode
    # 空キーは除外
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
        raise ValueError("meta に drive_file_id 列がありません。drive_file_id付きの meta に差し替えてください。")

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
st.set_page_config(page_title="Music Recommender (Experiment)", layout="wide")
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

        # query track_id
        st.session_state["query_track_id"] = QUERY_TRACKS.get(assigned, "")

        # base_idx を track_id で検索
        base_idx: Optional[int] = None
        qid = normalize_track_id(st.session_state["query_track_id"])
        if qid and "track_id" in meta.columns:
            mask = meta["track_id"].astype(str).map(normalize_track_id) == qid
            if mask.any():
                base_idx = int(np.argmax(mask.values))

        if base_idx is None:
            # 見つからない場合はランダム（実験では避けたいので、track_id設定を見直すこと）
            rng_tmp = np.random.default_rng(RANDOM_SEED)
            base_idx = int(rng_tmp.integers(0, len(V)))
            st.session_state["query_track_id"] = str(meta.loc[base_idx].get("track_id", ""))

        st.session_state["base_idx"] = base_idx

        st.session_state["phase"] = 1
        st.session_state["init_time"] = time.time()
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

# Sidebar
with st.sidebar:
    st.header("📊 管理者情報")
    st.write(f"Index rows: {info['index_rows']}")
    st.write(f"Embedding dim: {info['dim']}")
    st.write(f"requests available: {bool(info['has_requests'])}")
    st.write("---")
    st.write(f"Assigned genre: {st.session_state.get('assigned_genre')}")
    st.write(f"Session ID: {st.session_state.get('session_id')}")
    st.write(f"Query track_id: {st.session_state.get('query_track_id')}")
    st.write("---")
    c = load_counters()
    st.write("Completed counters:")
    for g in PRIMARY_GENRES + [RESERVE_GENRE]:
        limit = PRIMARY_LIMIT if g in PRIMARY_GENRES else POP_LIMIT
        st.write(f"- {g}: {c.get(g,0)}/{limit}")

# Phase update by time
now = time.time()
if st.session_state.get("phase") == 1 and now - st.session_state.get("init_time", 0) >= 15:
    st.session_state["phase"] = 2
if st.session_state.get("phase") == 3 and st.session_state.get("step3_time") is not None:
    if now - st.session_state["step3_time"] >= 20:
        st.session_state["phase"] = 4

###############################################################################
# Step1: Query
###############################################################################
guide("① まず、基準となる曲を聴いてください（30秒ほど推奨）。", st.session_state.get("phase") == 1)

base_idx = int(st.session_state["base_idx"])
base_row = meta.iloc[base_idx]

colL, colR = st.columns([1.2, 1.0])
with colL:
    st.subheader("🎵 基準曲 (Query Track)")
    title = str(base_row.get("title", "") or "")
    artist = str(base_row.get("artist", "") or "")
    if title or artist:
        st.markdown(f"**{title}**  —  {artist}".strip())
    else:
        st.markdown(f"**track_id={base_row.get('track_id','')}** / index={base_idx}")

with colR:
    st.markdown(f"**ジャンル:** {st.session_state.get('assigned_genre','')}")

def play_by_meta_row(row: pd.Series) -> None:
    file_id = str(row.get("drive_file_id", "") or "").strip()
    if not file_id:
        st.warning("⚠️ drive_file_id が空です（meta結合を確認）。")
        st.caption(f"debug: track_id={row.get('track_id','')}, idx={row.name}")
        return
    try:
        audio_bytes = download_mp3_bytes_from_drive(file_id)
        st.audio(audio_bytes, format="audio/mp3")
    except Exception as e:
        st.warning("⚠️ Drive から音源取得に失敗しました。共有設定を確認してください。")
        st.caption(f"debug: drive_file_id={file_id} / error={e}")

play_by_meta_row(base_row)

###############################################################################
# Step2: Top5
###############################################################################
guide("② 聴き終えたら、下のボタンを押してTop5推薦を表示してください。", st.session_state.get("phase") == 2)

colA, colB = st.columns([0.35, 0.65])
with colA:
    topk = st.number_input("Top-K", min_value=1, max_value=20, value=TOPK_DEFAULT, step=1)
with colB:
    run = st.button("🔎 この曲でTop5曲を表示", type="primary", disabled=(st.session_state.get("phase") < 2))

if run:
    q_vec = V[base_idx]
    idx_arr, sim_arr = cosine_topk(V, q_vec, topk=int(topk), exclude_idx=base_idx)

    st.session_state["topk_idx"] = idx_arr
    st.session_state["topk_sim"] = sim_arr
    st.session_state["phase"] = 3
    st.session_state["step3_time"] = time.time()

    rng = st.session_state["rng"]
    st.session_state["shuffle_order"] = rng.permutation(len(idx_arr))

    st.rerun()

###############################################################################
# Step3: Results
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

    letters = [chr(ord("A") + i) for i in range(len(rows_disp))]

    guide("③ 表示されたTop5を聴いて、好みに合うか評価してください。", st.session_state.get("phase") == 3)

    st.dataframe(pd.DataFrame([{
        "Position": L,
        "Title": r["title"],
        "Artist": r["artist"],
    } for L, r in zip(letters, rows_disp)]), use_container_width=True)

    # PCA
    st.markdown("### 🗺️ 推薦地図（PCA 2D / embeddings）")
    try:
        X_map = np.vstack([V[base_idx][None, :], V[idx_arr]])
        Z = pca2d_svd(X_map)
        Z = Z - Z[0]

        fig, ax = plt.subplots(figsize=(4.0, 2.4), dpi=120)
        ax.scatter(Z[1:, 0], Z[1:, 1], s=12)
        ax.scatter([0], [0], s=40, marker="*")
        for rnk, (xv, yv) in enumerate(Z[1:], start=1):
            ax.text(xv, yv, str(rnk), fontsize=8)
        ax.set_title("Recommendation Map (PCA 2D)", fontsize=8)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, clear_figure=True)
    except Exception as e:
        st.warning(f"PCA生成失敗: {e}")

    # Audio
    st.markdown("### 🎧 推薦曲プレビュー（シャッフル順）")
    for L, r in zip(letters, rows_disp):
        header = f"{L}"
        if r["title"]:
            header += f": {r['title']}"
        if r["artist"]:
            header += f" — {r['artist']}"
        st.markdown(f"**{header}**")
        play_by_meta_row(meta.iloc[int(r["index"])])

    # Prefill URL
    if st.session_state.get("prefilled_url") is None:
        params: Dict[str, str] = {}
        sid = st.session_state.get("session_id", "") or ""
        params[FORM_ENTRY_IDS.get("session_id", "")] = sid
        params[FORM_ENTRY_IDS.get("assigned_genre", "")] = st.session_state.get("assigned_genre", "") or ""
        params[FORM_ENTRY_IDS.get("query_track_id", "")] = str(st.session_state.get("query_track_id", "") or "")

        for pos, r in enumerate(rows_true, start=1):
            params[FORM_ENTRY_IDS.get(f"rec_track_id_{pos}", "")] = str(r["track_id"])
            params[FORM_ENTRY_IDS.get(f"true_rank_{pos}", "")] = str(r["true_rank"])
            params[FORM_ENTRY_IDS.get(f"sim_{pos}", "")] = str(r["similarity"])

        shuffle_pos_map: Dict[int, int] = {}
        if order is not None:
            for disp_pos, true_index in enumerate(order.tolist(), start=1):
                shuffle_pos_map[true_index] = disp_pos
        for pos in range(1, len(rows_true) + 1):
            params[FORM_ENTRY_IDS.get(f"shuffle_pos_{pos}", "")] = str(shuffle_pos_map.get(pos - 1, pos))

        url = build_prefilled_form_url(GOOGLE_FORM_URL, params)
        st.session_state["prefilled_url"] = url

        try:
            log_session_csv(sid, st.session_state.get("assigned_genre", ""), time.time(), url)
        except Exception as e:
            st.warning(f"ログ保存失敗: {e}")

    # Step4
    guide("④ 最後にアンケート（Googleフォーム）に回答してください。", st.session_state.get("phase") == 4)

    if st.session_state.get("phase") >= 4 and st.session_state.get("prefilled_url"):
        st.link_button("✅ Googleフォームへ進む", st.session_state["prefilled_url"])

        if not st.session_state.get("completed"):
            if st.button("🎉 完了 (クリックして参加を完了)"):
                sid = st.session_state.get("session_id", "")
                completed = load_completed()

                # 二重加算防止
                if sid in completed:
                    st.session_state["completed"] = True
                    st.info("このセッションは既に完了済みです。")
                else:
                    counters = load_counters()
                    g = st.session_state.get("assigned_genre", "")
                    limit = PRIMARY_LIMIT if g in PRIMARY_GENRES else POP_LIMIT

                    # 念のため上限チェック
                    if counters.get(g, 0) >= limit:
                        st.warning("このジャンルは上限に達しています（保存は行いません）。")
                    else:
                        counters[g] = int(counters.get(g, 0)) + 1
                        completed.add(sid)
                        try:
                            save_counters(counters)
                            save_completed(completed)
                        except Exception as e:
                            st.warning(f"保存に失敗: {e}")

                    st.session_state["completed"] = True
                    st.success("ご参加ありがとうございました。回答が記録されました。ブラウザを閉じて構いません。")
        else:
            st.success("このセッションは既に完了しました。ありがとうございました！")
    else:
        st.caption("※ 上のカウントダウン後にアンケートリンクが表示されます。")
else:
    st.caption("※ まだ推薦を実行していません。上の「この曲でTop5曲を表示」を押してください。")
