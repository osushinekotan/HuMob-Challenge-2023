import json
import shutil
from pathlib import Path

import pandas as pd
import yaml

from src.logger import Logger

logger = Logger()


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


def read_parquet_from_csv(filepath: Path, dirpath: Path) -> pd.DataFrame:
    name = filepath.name.split(".")[0]
    parquet_filepath = dirpath / f"{name}.parquet"
    if parquet_filepath.is_file():
        logger.info("load parquet file")
        return pd.read_parquet(parquet_filepath)

    logger.info("load csv & convert to parquet")
    df = pd.read_csv(filepath)
    df.to_parquet(parquet_filepath)
    return df
