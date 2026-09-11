"""记录并校验基础镜像中的 PyTorch 配套包，防止安装时被替换。"""
import argparse
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def installed_constraints() -> str:
    result = []
    for name in ("torch", "torchvision", "torchaudio", "triton"):
        try:
            result.append(f"{name}=={version(name)}")
        except PackageNotFoundError:
            if name == "torch":
                raise RuntimeError("基础镜像未安装 torch")
    return "\n".join(result) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    current = installed_constraints()
    if args.verify:
        # 新增配套包允许存在，但原有包必须保持相同版本。
        if not set(args.path.read_text().splitlines()) <= set(current.splitlines()):
            raise RuntimeError("基础镜像的 PyTorch 配套包被修改")
    else:
        args.path.write_text(current)
