import json
import shutil
from pathlib import Path

import yaml


def load_yaml(filepath: Path) -> dict:
    with open(filepath) as f:
        return yaml.safe_load(f)


def unzip(zip_filepath: Path, extract_dir: Path | None = None) -> None:
    if extract_dir is None:
        extract_dir = Path(zip_filepath.parent)
    output_dir = extract_dir / zip_filepath.stem
    if not output_dir.is_dir():
        shutil.unpack_archive(zip_filepath, extract_dir)


def save_json(filepath: Path, json_file: dict) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(
            json_file,
            f,
            indent=4,
            sort_keys=True,
            ensure_ascii=False,
        )


def load_json(filepath: Path) -> dict:
    with open(filepath) as f:
        dic = json.load(f)
    return dic
