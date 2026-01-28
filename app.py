# -*- coding: utf-8 -*-
"""
Experiment App (Streamlit local/Community Cloud) - Drive Playback + pop reserve + Built-in Questionnaire

UIは維持したまま、保存先だけを自動切替：
- ローカル（Secretsなし）: CSV + json（従来どおり）
- Streamlit Community Cloud（Secretsあり）: Google Sheets（responses/sessions）

★追加（表示のみ）:
- 基準曲の再生の上に「30秒程度推奨」
- 推薦曲の再生のところに「15秒ほど推奨」

## 改善点

このバージョンでは以下の点を改善しました。

* **session_id の登録タイミングを変更**  
  これまではセッション開始時に `reserve_session_sheets()` または `append_session_log_local()`
  を呼び出していましたが、アンケートを開始しただけで session_id が記録されてしまう問題がありました。  
  この版では、セッション ID の登録およびステータス更新は完了ボタンを押したタイミングで行われます。初期化時にはジャンル割当てのみを行い、実際の登録は完了時に実施します。

* **推薦曲の下に順位入力欄を配置**  
  A〜E の各推薦曲の再生ボタンの下に「1〜5」から選べる順位入力欄を設け、より直感的に順位を選択できるようにしました。  
  入力された順位は従来と同様に重複チェックを行い、未選択や重複がある場合は警告を表示します。

* **基本情報（年齢・性別・音楽視聴時間）の入力を末尾に移動**  
  ユーザー属性に関する質問をアンケートの最後にまとめ、その他の回答項目の後に表示するよう変更しました。

* **ローカル保存の場所を明示**  
  Google Sheets への書き込みに失敗した場合でもデータを捨てず、
  回答内容は `data/responses.csv` に、セッション情報は `data/sessions_log.csv` に保存されます。

このファイルは、元の app.py の構造や変数名を極力保ちつつ上記の修正を適用したものです。
"""

from __future__ import annotations

import csv
import json
import re
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    import requests  # type: ignore
    _HAS_REQUESTS = True
except Exception:
    _HAS_REQUESTS = False
    import urllib.request

# Sheets用
try:
    import gspread  # type: ignore
    from google.oauth2.service_account import Credentials  # type: ignore
    _HAS_SHEETS_LIBS = True
except Exception:
    _HAS_SHEETS_LIBS = False


###############################################################################
# CONFIG
###############################################################################
BASE_DIR = Path(__file__).resolve().parent

INDEX_DIR = BASE_DIR / "index"
EMB_NPY_NAME = "index_emb64_l2.npy"
EMB_META_NAME = "index_emb64_l2_meta.csv"

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESPONSES_CSV = DATA_DIR / "responses.csv"
SESSIONS_LOG_CSV = DATA_DIR / "sessions_log.csv"

TOPK_FIXED = 5
RANDOM_SEED = None

# 「間隔を広げる」：真順位 1,5,10,15,20 を選ぶ（0-indexなら 0,4,9,14,19）
# 全ジャンル共通で固定
SPACED_RANK_POSITIONS = [0, 19, 39, 59, 79]

PRIMARY_GENRES: List[str] = ["classical", "jazz", "rock", "hiphop"]
RESERVE_GENRE: str = "pop"
PRIMARY_ORDER: List[str] = ["classical", "jazz", "rock", "hiphop"]
PRIMARY_LIMIT: int = 10
POP_LIMIT: int = 10

QUERY_TRACKS: Dict[str, str] = {
    "classical": "1155894",
    "jazz": "1069786",
    "rock": "1134644",
    "hiphop": "1157595",
    "pop": "1030923",
}

# ローカル用
COUNTERS_FILE = BASE_DIR / "genre_counter.json"
COMPLETED_FILE = BASE_DIR / "completed_sessions.json"

LETTERS = ["A", "B", "C", "D", "E"]
FEATURES = ["tempo", "rhythm", "vocal", "melody", "genre_sim"]

LIKERT_MIN = 1
LIKERT_MAX = 5

MUSIC_HOURS_OPTIONS = ["全く聴かない", "３０分未満", "１時間", "２時間以上"]
AGE_GROUP_OPTIONS = ["１０代", "２０代", "３０代", "４０代以上"]
GENDER_OPTIONS = ["未回答", "男性", "女性", "その他", "回答しない"]

RESERVATION_TTL_SEC = 60 * 60  # 1時間


###############################################################################
# Helpers
###############################################################################
def l2_unit(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = float(np.linalg.norm(x))
    return x / (n + eps)

def normalize_track_id(x: object) -> str:
    if x is None:
        return ""
    return Path(str(x).strip()).stem

def new_session_id() -> str:
    return uuid.uuid4().hex

def _retry(fn, tries: int = 3, base_sleep: float = 0.6):
    """
    Sheets通信の一時失敗対策：軽いリトライ
    """
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            time.sleep(base_sleep * (2 ** i))
    raise last  # type: ignore

def cosine_spaced_pick(
    V: np.ndarray,
    q: np.ndarray,
    topk: int,
    positions: List[int],
    exclude_idx: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    類似度上位から「指定順位（positions）」を間引いて選ぶ。
    例：positions=[0,4,9,14,19] -> 真順位 1,5,10,15,20

    戻り値：
    - idx_pick: 選ばれたインデックス（len=topk）
    - sim_pick: その類似度
    - rank_pick: 真順位（1-index）
    """
    sims = V @ q
    if exclude_idx is not None and 0 <= exclude_idx < len(sims):
        sims[exclude_idx] = -1e9

    order = np.argsort(-sims)  # 真順位（0-index）
    pick_pos = [p for p in positions if 0 <= p < len(order)]

    # 5個取れない場合の保険：最後のpos以降から順に埋める
    if len(pick_pos) < topk:
        start = (pick_pos[-1] + 1) if pick_pos else 0
        for p in range(start, len(order)):
            if p not in pick_pos:
                pick_pos.append(p)
            if len(pick_pos) >= topk:
                break

    pick_pos = pick_pos[:topk]
    idx_pick = np.array([int(order[p]) for p in pick_pos], dtype=int)
    sim_pick = np.array([float(sims[idx]) for idx in idx_pick], dtype=float)
    rank_pick = np.array([int(p + 1) for p in pick_pos], dtype=int)  # 1-index
    return idx_pick, sim_pick, rank_pick


###############################################################################
# Local persistence (json/csv)
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

def load_counters_local() -> Dict[str, int]:
    data = _load_json(COUNTERS_FILE, default={})
    for g in PRIMARY_GENRES + [RESERVE_GENRE]:
        data[g] = int(data.get(g, 0))
    return data

def save_counters_local(counters: Dict[str, int]) -> None:
    _save_json_atomic(COUNTERS_FILE, counters)

def load_completed_local() -> set[str]:
    data = _load_json(COMPLETED_FILE, default={"completed": []})
    return set(data.get("completed", []))

def save_completed_local(completed: set[str]) -> None:
    _save_json_atomic(COMPLETED_FILE, {"completed": sorted(list(completed))})

def assign_genre_pop_reserve_from_counts(counts: Dict[str, int]) -> Optional[str]:
    for g in PRIMARY_ORDER:
        if counts.get(g, 0) < PRIMARY_LIMIT:
            return g
    if counts.get(RESERVE_GENRE, 0) < POP_LIMIT:
        return RESERVE_GENRE
    return None

def append_session_log_local(session_id: str, assigned_genre: str, query_track_id: str) -> None:
    SESSIONS_LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": int(time.time()),
        "session_id": session_id,
        "assigned_genre": assigned_genre,
        "query_track_id": query_track_id,
    }
    write_header = not SESSIONS_LOG_CSV.exists()
    with SESSIONS_LOG_CSV.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


###############################################################################
# Response schema (shared)
###############################################################################
def response_schema_columns() -> List[str]:
    cols = []
    cols += ["timestamp", "session_id", "assigned_genre", "query_track_id"]
    cols += ["music_hours_per_day", "age", "gender"]
    cols += ["A_track_id", "B_track_id", "C_track_id", "D_track_id", "E_track_id"]
    cols += ["A_true_rank", "B_true_rank", "C_true_rank", "D_true_rank", "E_true_rank"]
    cols += ["A_sim", "B_sim", "C_sim", "D_sim", "E_sim"]
    cols += ["rank_A", "rank_B", "rank_C", "rank_D", "rank_E"]
    for L in LETTERS:
        for feat in FEATURES:
            cols.append(f"{L}_{feat}")
    cols += ["usability", "ui_visibility", "free_comment"]
    return cols

def append_response_row_local(row: Dict[str, object]) -> None:
    cols = response_schema_columns()
    write_header = not RESPONSES_CSV.exists()
    with RESPONSES_CSV.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if write_header:
            w.writeheader()
        out = {c: row.get(c, "") for c in cols}
        w.writerow(out)


###############################################################################
# Google Sheets backend (GCP_SA_JSON 対応) + 安定化
###############################################################################
def using_sheets_backend() -> bool:
    return (
        _HAS_SHEETS_LIBS and
        ("SHEET_ID" in st.secrets) and
        ("GCP_SA_JSON" in st.secrets)
    )

@st.cache_resource(show_spinner=False)
def get_gspread_client() -> "gspread.Client":
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    sa_json_str = st.secrets["GCP_SA_JSON"]
    sa_info = json.loads(sa_json_str)
    creds = Credentials.from_service_account_info(sa_info, scopes=scopes)
    return gspread.authorize(creds)

@st.cache_resource(show_spinner=False)
def get_sheet_cached() -> "gspread.Spreadsheet":
    sheet_id = st.secrets["SHEET_ID"]
    return get_gspread_client().open_by_key(sheet_id)

def ws(name: str) -> "gspread.Worksheet":
    return get_sheet_cached().worksheet(name)

def ensure_sheet_headers():
    def _do():
        wsr = ws("responses")
        wss = ws("sessions")

        resp_cols = response_schema_columns()
        sess_cols = ["timestamp", "session_id", "status", "assigned_genre", "query_track_id"]  # reserved/completed

        vals = wsr.get_all_values()
        if not vals:
            wsr.append_row(resp_cols, value_input_option="RAW")
        else:
            if len(vals[0]) == 0:
                wsr.update("A1", [resp_cols])

        vals2 = wss.get_all_values()
        if not vals2:
            wss.append_row(sess_cols, value_input_option="RAW")
        else:
            if len(vals2[0]) == 0:
                wss.update("A1", [sess_cols])
    _retry(_do, tries=3, base_sleep=0.6)

def count_sessions_by_genre_sheets(now_ts: int) -> Dict[str, int]:
    def _do():
        wss = ws("sessions")
        recs = wss.get_all_records()
        counts = {g: 0 for g in PRIMARY_GENRES + [RESERVE_GENRE]}
        for r in recs:
            g = str(r.get("assigned_genre", "")).strip()
            status = str(r.get("status", "")).strip()
            ts = r.get("timestamp", None)
            try:
                ts = int(ts)
            except Exception:
                ts = None

            if g not in counts:
                continue

            if status == "completed":
                counts[g] += 1
            elif status == "reserved":
                if ts is not None and (now_ts - ts) <= RESERVATION_TTL_SEC:
                    counts[g] += 1
        return counts
    return _retry(_do, tries=3, base_sleep=0.6)

def reserve_session_sheets(now_ts: int, session_id: str, assigned_genre: str, query_track_id: str) -> None:
    """
    セッションの予約をシートに書き込む。status='reserved'。
    完了時には mark_completed_sheets() で 'completed' に更新する。
    """
    def _do():
        wss = ws("sessions")
        wss.append_row([now_ts, session_id, "reserved", assigned_genre, query_track_id], value_input_option="RAW")
    _retry(_do, tries=3, base_sleep=0.6)

def mark_completed_sheets(session_id: str) -> None:
    def _do():
        wss = ws("sessions")
        cell = wss.find(session_id)
        if cell is None:
            return
        wss.update_cell(cell.row, 3, "completed")  # status列
    _retry(_do, tries=3, base_sleep=0.6)

def is_completed_sheets(session_id: str) -> bool:
    def _do():
        wss = ws("sessions")
        cell = wss.find(session_id)
        if cell is None:
            return False
        status = wss.cell(cell.row, 3).value
        return str(status).strip() == "completed"
    return bool(_retry(_do, tries=3, base_sleep=0.6))

def append_response_row_sheets(row: Dict[str, object]) -> None:
    def _do():
        wsr = ws("responses")
        cols = response_schema_columns()
        values = [row.get(c, "") for c in cols]
        wsr.append_row(values, value_input_option="RAW")
    _retry(_do, tries=3, base_sleep=0.6)


###############################################################################
# Google Drive download (audio)
###############################################################################
def drive_download_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}"

def _download_bytes_via_requests(url: str, timeout: int = 30) -> bytes:
    sess = requests.Session()
    r = sess.get(url, stream=True, timeout=timeout)
    r.raise_for_status()
    ctype = (r.headers.get("content-type") or "").lower()
    if "text/html" in ctype:
        html = r.text
        m = re.search(r"confirm=([0-9A-Za-z_]+)", html)
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

@st.cache_data(show_spinner=False, ttl=60 * 60, max_entries=800)
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
    info = {"index_rows": int(len(V)), "dim": int(V.shape[1]) if len(V) > 0 else 0, "has_requests": int(_HAS_REQUESTS)}
    return V, meta, info


###############################################################################
# UI
###############################################################################
st.set_page_config(page_title="🎧 Music Recommender (Experiment)", layout="wide", initial_sidebar_state="collapsed")
st.title("🎧 Music Recommender – Experiment")

st.markdown(
    """
### このページで行うこと
1. **基準曲**を聴く  
2. **Top5（A〜E）**を表示して聴く（表示順はランキング順ではありません）  
3. **推薦曲 A〜E の類似度順位を入力**  
4. **各曲の類似度アンケートに回答**  
5. **アプリ全体に関する評価と自由記述**  
6. **基本情報（年齢・性別・音楽視聴時間）を入力**  
7. **完了**を押して送信（この時点で保存・カウントされます）
"""
)

# Load assets
V, meta, info = load_assets()

# Detect backend
USE_SHEETS = using_sheets_backend()

# ---- 安定化：Sheets初期化は「起動時に1回だけ」＋失敗しても止めない ----
if USE_SHEETS and ("sheets_ready" not in st.session_state):
    try:
        ensure_sheet_headers()
        st.session_state["sheets_ready"] = True
    except Exception:
        st.session_state["sheets_ready"] = False

# 失敗してもUIは止めない（保存時にフォールバック）
if USE_SHEETS and not st.session_state.get("sheets_ready", False):
    st.warning("現在、Google Sheets への接続が不安定です。回答は続けられますが、保存時に失敗する可能性があります。")

# Sidebar admin
with st.sidebar:
    with st.expander("管理者情報（クリックで展開）", expanded=False):
        backend_name = "Google Sheets" if (USE_SHEETS and st.session_state.get("sheets_ready", False)) else "Local files"
        st.write(f"Backend: {backend_name}")
        st.write(f"Index rows: {info['index_rows']}")
        st.write(f"Embedding dim: {info['dim']}")
        if USE_SHEETS and st.session_state.get("sheets_ready", False):
            st.write("Sheets: responses / sessions")
        else:
            st.write(f"responses.csv: {RESPONSES_CSV}")

# Session init
if "initialised" not in st.session_state:
    st.session_state["initialised"] = True

    now_ts = int(time.time())
    session_id = new_session_id()

    use_sheets_now = USE_SHEETS and st.session_state.get("sheets_ready", False)

    if use_sheets_now:
        try:
            counts = count_sessions_by_genre_sheets(now_ts)
            assigned = assign_genre_pop_reserve_from_counts(counts)
            if assigned is None:
                st.session_state["closed"] = True
            else:
                st.session_state["closed"] = False
                st.session_state["assigned_genre"] = assigned
                st.session_state["session_id"] = session_id
                st.session_state["completed"] = False
                st.session_state["query_track_id"] = QUERY_TRACKS.get(assigned, "")
                # セッションの予約は完了ボタン押下時に行うため、ここでは記録しない
        except Exception:
            # Sheetsが途中で落ちたらローカルへ
            st.session_state["sheets_ready"] = False
            counters = load_counters_local()
            assigned = assign_genre_pop_reserve_from_counts(counters)
            if assigned is None:
                st.session_state["closed"] = True
            else:
                st.session_state["closed"] = False
                st.session_state["assigned_genre"] = assigned
                st.session_state["session_id"] = session_id
                st.session_state["completed"] = False
                st.session_state["query_track_id"] = QUERY_TRACKS.get(assigned, "")
                # ローカルでも予約登録は完了ボタンまで遅延
    else:
        counters = load_counters_local()
        assigned = assign_genre_pop_reserve_from_counts(counters)
        if assigned is None:
            st.session_state["closed"] = True
        else:
            st.session_state["closed"] = False
            st.session_state["assigned_genre"] = assigned
            st.session_state["session_id"] = session_id
            st.session_state["completed"] = False
            st.session_state["query_track_id"] = QUERY_TRACKS.get(assigned, "")
            # ローカルでも予約登録は完了ボタンまで遅延

    # 初期化：ベース曲のインデックスを決定
    if not st.session_state.get("closed"):
        base_idx: Optional[int] = None
        qid = normalize_track_id(st.session_state["query_track_id"])
        if qid and "track_id" in meta.columns:
            mask = meta["track_id"].astype(str).map(normalize_track_id) == qid
            if mask.any():
                base_idx = int(np.argmax(mask.values))
        if base_idx is None:
            rng_tmp = np.random.default_rng(RANDOM_SEED)
            base_idx = int(rng_tmp.integers(0, len(V)))
            st.session_state["query_track_id"] = str(meta.loc[base_idx].get("track_id", ""))
        st.session_state["base_idx"] = base_idx

        st.session_state["topk_idx"] = None
        st.session_state["topk_sim"] = None
        st.session_state["topk_true_rank"] = None
        st.session_state["shuffle_order"] = None

        # 初期値：順位や評価を初期化
        for L in LETTERS:
            st.session_state[f"rank_{L}"] = None
        for L in LETTERS:
            for feat in FEATURES:
                st.session_state[f"{L}_{feat}"] = 3
        # 基本情報を初期化
        st.session_state["music_hours_per_day"] = MUSIC_HOURS_OPTIONS[0]
        st.session_state["age"] = AGE_GROUP_OPTIONS[1]
        st.session_state["gender"] = "未回答"
        # アプリ評価
        st.session_state["usability"] = 3
        st.session_state["ui_visibility"] = 3
        st.session_state["free_comment"] = ""

if st.session_state.get("closed"):
    st.error("全ジャンルが満員です（実験終了）。")
    st.stop()

# Playback helper
def play_by_meta_row(row: pd.Series) -> None:
    file_id = str(row.get("drive_file_id", "") or "").strip()
    if not file_id:
        st.warning("⚠️ 音源が再生できません（drive_file_id が空です）。")
        return
    audio_bytes = download_mp3_bytes_from_drive(file_id)
    st.audio(audio_bytes, format="audio/mp3")

# Step 1
st.markdown("## ① 基準曲")
st.caption("※ 30秒程度の試聴を推奨します。バーを動かして飛ばしながら聴いていただいても構いません。")
base_idx = int(st.session_state["base_idx"])
base_row = meta.iloc[base_idx]
title = str(base_row.get("title", "") or "")
artist = str(base_row.get("artist", "") or "")
if title or artist:
    st.markdown(f"**{title}**  —  {artist}".strip())
else:
    st.markdown(f"**track_id={base_row.get('track_id','')}**")
play_by_meta_row(base_row)

# Step 2
st.markdown("## ② Top5を表示")
run = st.button("🔎 この曲から5つの楽曲を表示", type="primary")
if run:
    q_vec = V[base_idx]
    idx_arr, sim_arr, true_rank_arr = cosine_spaced_pick(
        V, q_vec,
        topk=TOPK_FIXED,
        positions=SPACED_RANK_POSITIONS,
        exclude_idx=base_idx
    )
    st.session_state["topk_idx"] = idx_arr
    st.session_state["topk_sim"] = sim_arr
    st.session_state["topk_true_rank"] = true_rank_arr
    rng = np.random.default_rng(RANDOM_SEED)
    st.session_state["shuffle_order"] = rng.permutation(len(idx_arr))
    for L in LETTERS:
        st.session_state[f"rank_{L}"] = None
    st.rerun()

# Step 3 onward: show recommendations and collect survey
if st.session_state.get("topk_idx") is not None:
    idx_arr = st.session_state["topk_idx"]
    sim_arr = st.session_state["topk_sim"]
    rank_arr = st.session_state["topk_true_rank"]
    order = st.session_state.get("shuffle_order")

    # prepare mapping of true rows
    rows_true: List[dict] = []
    for (i_val, s_val, rnk) in zip(idx_arr, sim_arr, rank_arr):
        i_int = int(i_val)
        r = meta.iloc[i_int]
        rows_true.append({
            "true_rank": int(rnk),  # 真順位（1,5,10,15,20 など）
            "similarity": float(np.round(float(s_val), 6)),
            "index": i_int,
            "track_id": str(r.get("track_id","")),
            "title": str(r.get("title","") or ""),
            "artist": str(r.get("artist","") or ""),
        })

    rows_disp = [rows_true[i] for i in order] if order is not None else rows_true
    disp_map = {L: row for L, row in zip(LETTERS, rows_disp)}

    # Step 3: Recommend tracks with ranking input below each
    st.markdown("## ③ 推薦曲（A〜E）")
    st.caption("A〜Eの表示順はランキング順ではありません。")
    for L in LETTERS:
        r = disp_map[L]
        header = f"{L}"
        if r["title"]:
            header += f": {r['title']}"
        if r["artist"]:
            header += f" — {r['artist']}"
        st.markdown(f"### {header}")
        st.caption("※ 15秒ほどの試聴を推奨します")
        # audio playback
        play_by_meta_row(meta.iloc[int(r["index"])] )
        # ranking input below audio
        rank_key = f"rank_{L}"
        # prepare options and current value
        rank_options = ["未選択", "1", "2", "3", "4", "5"]
        current_rank = st.session_state.get(rank_key)
        idx_sel = 0
        if isinstance(current_rank, int) and 1 <= current_rank <= 5:
            try:
                idx_sel = rank_options.index(str(current_rank))
            except ValueError:
                idx_sel = 0
        sel = st.selectbox(
            f"{L} の類似度順位を選択",
            options=rank_options,
            index=idx_sel,
            key=f"ui_{rank_key}"
        )
        st.session_state[rank_key] = None if sel == "未選択" else int(sel)

    # Validate ranking selections
    ranks = [st.session_state.get(f"rank_{L}") for L in LETTERS]
    all_selected = all(isinstance(v, int) for v in ranks)
    no_dup = (len(set(ranks)) == 5) if all_selected else False
    if not all_selected:
        st.warning("類似度順位が未選択の項目があります。A〜Eすべて選択してください。")
    elif not no_dup:
        st.error("順位が重複しています。1〜5がそれぞれ一度ずつになるように修正してください。")
    else:
        st.success("順位の入力はOKです。")

    # Step 4: Similarity evaluation
    st.markdown("## ④ 類似度評価（1=似ていない ～ 5=とても似ている）")
    label_map = {
        "tempo": "テンポ（速さ）",
        "rhythm": "リズム（ノリ）",
        "vocal": "ボーカル（声色）",
        "melody": "メロディ",
        "genre_sim": "ジャンル",
    }
    for L in LETTERS:
        with st.expander(f"{L} の評価", expanded=False):
            for feat in FEATURES:
                k = f"{L}_{feat}"
                st.session_state[k] = st.slider(
                    f"{label_map[feat]}の類似度",
                    min_value=LIKERT_MIN, max_value=LIKERT_MAX,
                    value=int(st.session_state.get(k, 3)),
                    step=1, key=f"ui_{k}"
                )

    # Step 5: App evaluation and free comment
    st.markdown("## ⑤ アプリについて")
    st.session_state["usability"] = st.slider(
        "アプリの使いやすさ（1=悪い ～ 5=良い）",
        min_value=LIKERT_MIN, max_value=LIKERT_MAX,
        value=int(st.session_state.get("usability", 3)),
        step=1, key="ui_usability"
    )
    st.session_state["ui_visibility"] = st.slider(
        "UIの見やすさ（1=悪い ～ 5=良い）",
        min_value=LIKERT_MIN, max_value=LIKERT_MAX,
        value=int(st.session_state.get("ui_visibility", 3)),
        step=1, key="ui_ui_visibility"
    )
    st.session_state["free_comment"] = st.text_area(
        "本アプリについてご意見をお聞かせください（自由記述）",
        value=str(st.session_state.get("free_comment","")),
        key="ui_free_comment",
        height=120
    )

    # Step 6: Basic demographic information moved to the end
    st.markdown("## ⑥ 基本情報（年齢・性別・音楽視聴時間）")
    # We still group these in an expander for compactness
    with st.expander("基本情報を入力", expanded=True):
        st.session_state["music_hours_per_day"] = st.radio(
            "1日にどれくらい音楽を聴きますか",
            options=MUSIC_HOURS_OPTIONS,
            index=MUSIC_HOURS_OPTIONS.index(st.session_state.get("music_hours_per_day", MUSIC_HOURS_OPTIONS[0])),
            horizontal=True,
        )
        st.session_state["age"] = st.radio(
            "年齢",
            options=AGE_GROUP_OPTIONS,
            index=AGE_GROUP_OPTIONS.index(st.session_state.get("age", AGE_GROUP_OPTIONS[1])),
            horizontal=True,
        )
        st.session_state["gender"] = st.selectbox(
            "性別",
            options=GENDER_OPTIONS,
            index=GENDER_OPTIONS.index(st.session_state.get("gender", "未回答")) if st.session_state.get("gender","未回答") in GENDER_OPTIONS else 0,
        )

    # Step 7: Completion section
    st.markdown("## ⑦ 完了")

    gender_ok = st.session_state.get("gender") not in (None, "", "未回答")
    demo_ok = (st.session_state.get("music_hours_per_day") in MUSIC_HOURS_OPTIONS and
               st.session_state.get("age") in AGE_GROUP_OPTIONS and gender_ok)
    can_submit = all_selected and no_dup and demo_ok and (not st.session_state.get("completed", False))

    if not gender_ok:
        st.warning("性別が「未回答」です。回答しない場合は「回答しない」を選んでください。")

    if st.session_state.get("completed"):
        st.success("このセッションは既に完了しています。ありがとうございました！")
    else:
        if st.button("🎉 完了（保存）", type="primary", disabled=(not can_submit)):
            sid = st.session_state.get("session_id", "") or ""
            now_ts = int(time.time())

            # ---- 保存（Sheets優先、失敗したらローカルにフォールバック） ----
            row: Dict[str, object] = {}
            row["timestamp"] = now_ts
            row["session_id"] = sid
            row["assigned_genre"] = st.session_state.get("assigned_genre", "")
            row["query_track_id"] = st.session_state.get("query_track_id", "")
            row["music_hours_per_day"] = st.session_state.get("music_hours_per_day", "")
            row["age"] = st.session_state.get("age", "")
            row["gender"] = st.session_state.get("gender", "")

            for L in LETTERS:
                row[f"{L}_track_id"] = str(disp_map[L]["track_id"])
                row[f"{L}_true_rank"] = int(disp_map[L]["true_rank"])
                row[f"{L}_sim"] = float(disp_map[L]["similarity"])
                row[f"rank_{L}"] = int(st.session_state.get(f"rank_{L}"))

            for L in LETTERS:
                for feat in FEATURES:
                    row[f"{L}_{feat}"] = int(st.session_state.get(f"{L}_{feat}", 3))

            row["usability"] = int(st.session_state.get("usability", 3))
            row["ui_visibility"] = int(st.session_state.get("ui_visibility", 3))
            row["free_comment"] = str(st.session_state.get("free_comment", ""))

            saved_to_sheets = False

            if USE_SHEETS and st.session_state.get("sheets_ready", False):
                try:
                    # 二重登録を避ける
                    if is_completed_sheets(sid):
                        st.session_state["completed"] = True
                        st.info("このセッションは既に完了済みです。")
                        st.stop()

                    # 予約してその場で完了マークを付ける
                    reserve_session_sheets(now_ts, sid, st.session_state.get("assigned_genre",""), st.session_state.get("query_track_id",""))
                    mark_completed_sheets(sid)
                    # 回答を保存
                    append_response_row_sheets(row)
                    saved_to_sheets = True
                except Exception:
                    saved_to_sheets = False
                    st.session_state["sheets_ready"] = False

            if not saved_to_sheets:
                # Sheetsに失敗してもデータを捨てない：ローカル保存
                append_response_row_local(row)
                # セッションログもローカルに保存
                append_session_log_local(sid, st.session_state.get("assigned_genre",""), st.session_state.get("query_track_id",""))
                completed_ids = load_completed_local()
                completed_ids.add(sid)
                save_completed_local(completed_ids)
                counters = load_counters_local()
                g = st.session_state.get("assigned_genre","")
                counters[g] = int(counters.get(g, 0)) + 1
                save_counters_local(counters)
                if USE_SHEETS:
                    st.warning("Google Sheets への保存に失敗したため、ローカルに保存しました（管理者に連絡してください）。")

            st.session_state["completed"] = True
            st.success("保存しました。ご協力ありがとうございました！ブラウザを閉じてください。。")

else:
    st.info("まず「🔎 この曲から5つの楽曲を表示」を押してください。")
