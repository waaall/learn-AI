"""检查本地 Qwen3-TTS 目录，提前发现缺失分片、断开的缓存链接和 LFS 指针。"""
import json
from pathlib import Path


def require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"模型文件缺失或为空：{path}")
    with path.open("rb") as stream:
        if stream.read(100).startswith(b"version https://git-lfs.github.com/spec/"):
            raise ValueError(f"文件只是 Git LFS 指针，尚未下载实际内容：{path}")


def check_weights(root: Path) -> None:
    # 兼容单文件及索引分片，不把特定模型的分片数量写死。
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = root / name
        if index.is_file():
            require_file(index)
            shards = set(json.loads(index.read_text())["weight_map"].values())
            if not shards:
                raise ValueError(f"权重索引为空：{index}")
            for shard in shards:
                relative = Path(shard)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"权重索引包含非法相对路径：{index}")
                require_file(root / relative)
            return
    for name in ("model.safetensors", "pytorch_model.bin"):
        if (root / name).is_file():
            require_file(root / name)
            return
    raise ValueError(f"未找到完整权重或分片索引：{root}")


def validate_model_directory(directory: str) -> Path:
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"本地模型目录不存在：{root}")
    for name in ("config.json", "generation_config.json", "tokenizer_config.json", "preprocessor_config.json",
                 "speech_tokenizer/config.json", "speech_tokenizer/preprocessor_config.json"):
        require_file(root / name)
    if json.loads((root / "config.json").read_text()).get("tts_model_type") != "custom_voice":
        raise ValueError("此服务需要 CustomVoice 模型目录")
    if (root / "tokenizer.json").is_file():
        require_file(root / "tokenizer.json")
    else:
        for name in ("vocab.json", "merges.txt"):
            require_file(root / name)
    check_weights(root)
    check_weights(root / "speech_tokenizer")
    return root
