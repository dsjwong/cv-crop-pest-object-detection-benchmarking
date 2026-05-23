import kagglehub
from pathlib import Path
import shutil

# Project root = 当前文件所在目录的父级目录
project_root = Path(__file__).resolve().parents[2]   # <- 关键
print("Project root:", project_root)

# Downloading dataset
path = kagglehub.dataset_download("rupankarmajumdar/crop-pests-dataset")
print("Path to dataset files:", path)

# Destination = project_root / datasets/raw
dst = project_root / "datasets/raw"
dst.mkdir(parents=True, exist_ok=True)

# Sync files
for p in Path(path).iterdir():
    target = dst / p.name
    if p.is_dir():
        if target.exists(): shutil.rmtree(target)
        shutil.copytree(p, target)
    else:
        shutil.copy2(p, target)

print("Raw dataset synced to:", dst.resolve())
