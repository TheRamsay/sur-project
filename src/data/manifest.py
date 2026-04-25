from pathlib import Path

_FOLDERS = ("target_train", "target_dev", "non_target_train", "non_target_dev")


def find_wav(stem: str, data_dir: Path) -> Path:
    for sf in _FOLDERS:
        p = data_dir / sf / (stem + ".wav")
        if p.exists():
            return p
    raise FileNotFoundError(stem)


def find_png(stem: str, data_dir: Path) -> Path:
    for sf in _FOLDERS:
        p = data_dir / sf / (stem + ".png")
        if p.exists():
            return p
    raise FileNotFoundError(stem)
