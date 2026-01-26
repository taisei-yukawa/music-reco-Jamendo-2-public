# Music Recommender – Experiment (Streamlit Community Cloud)

Jamendoベースの埋め込みインデックスを使った楽曲推薦システム（被験者評価用）です。

## Features
- ジャンル割当（primary 4ジャンル + pop予備枠）
- クエリ曲をジャンルごとに固定
- Top5を内部順位保持したまま表示順シャッフル（ランキング提示バイアス対策）
- Google Form へ prefilled URL で遷移（session_id 等を埋め込み）
- 完了ボタン押下でのみカウント（未完セッション無視 / 二重加算防止）

## Required files
app.py
util.py
requirements.txt
index/index_emb64_l2.npy
index/index_emb64_l2_meta.csv (drive_file_id列を含む)


## Google Drive audio
音源はGitHubに含めず、Google Drive の `drive_file_id` でダウンロードして再生します。

- mp3 ファイル（または上位フォルダ）は「リンクを知っている全員が閲覧可」にしてください。
- `index_emb64_l2_meta.csv` に `drive_file_id` が入っている必要があります。

## Setup (local)
```bash
pip install -r requirements.txt
streamlit run app.py
