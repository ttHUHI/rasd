"""tqdm + 文件日志"""
import sys
from pathlib import Path
from tqdm import tqdm


class Logger:
    def __init__(self, log_file=None):
        self.log_file = Path(log_file) if log_file else None
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            self._f = open(self.log_file, "w", encoding="utf-8")

    def write(self, msg):
        sys.stdout.write(msg)
        sys.stdout.flush()
        if self._f:
            self._f.write(msg)
            self._f.flush()

    def close(self):
        if self._f:
            self._f.close()


def progress(iterable, desc="", total=None):
    return tqdm(iterable, desc=desc, total=total, ncols=100, file=sys.stdout)
