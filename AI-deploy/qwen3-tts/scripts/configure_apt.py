"""调整 Ubuntu 镜像地址，同时兼容传统格式和 DEB822 格式。"""
import os
import re
from pathlib import Path


def configure(root: Path, mirror: str, security_mirror: str) -> None:
    files = [root / "sources.list"]
    files += list((root / "sources.list.d").glob("*.list"))
    files += list((root / "sources.list.d").glob("*.sources"))
    for path in files:
        if not path.is_file():
            continue
        text = path.read_text()
        # 只匹配官方 Ubuntu 地址，不误改 NVIDIA 等第三方仓库。
        text = re.sub(r"https?://(?:[\w-]+\.)?archive\.ubuntu\.com/ubuntu/?",
                      mirror.rstrip("/") + "/", text)
        text = re.sub(r"https?://security\.ubuntu\.com/ubuntu/?",
                      security_mirror.rstrip("/") + "/", text)
        path.write_text(text)


if __name__ == "__main__":
    configure(Path("/etc/apt"), os.environ["APT_MIRROR"],
              os.environ["APT_SECURITY_MIRROR"])
