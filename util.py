# util.py
# -*- coding: utf-8 -*-
"""
Minimal utilities for the Streamlit experiment app (pop reserve version).

この util.py は「pop別枠対応 app.py」がアプリ内で割当・カウント・完了判定を完結している前提の
“最小ユーティリティ”です。

- app.py 側で JSON 保存処理を持っているなら、この util.py すら不要です。
- ただし、今後ロジック分離したくなった場合に備えて、
  JSON の安全な読み書き（atomic save）だけを提供します。

注意:
- Streamlit Community Cloud では、保存先はコンテナ内です。
  再デプロイや環境更新で消える可能性があります（永続DBではありません）。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Set


def load_json(path: Path, default: Any) -> Any:
    """JSON を安全に読み込む。失敗したら default を返す。"""
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default


def save_json_atomic(path: Path, obj: Any) -> None:
    """
    JSON を atomic に保存する。
    - 書き込み途中で落ちても壊れないよう、tmp に書いてから置換する。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{int(time.time())}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_counters(path: Path, genres: list[str]) -> Dict[str, int]:
    """{genre: count} を読み込む。存在しない genre は 0 補完。"""
    data = load_json(path, default={})
    out: Dict[str, int] = {}
    for g in genres:
        out[g] = int(data.get(g, 0))
    return out


def save_counters(path: Path, counters: Dict[str, int]) -> None:
    save_json_atomic(path, counters)


def load_completed(path: Path) -> Set[str]:
    """{"completed":[...]} を set で返す。"""
    data = load_json(path, default={"completed": []})
    return set(data.get("completed", []))


def save_completed(path: Path, completed: Set[str]) -> None:
    save_json_atomic(path, {"completed": sorted(list(completed))})
