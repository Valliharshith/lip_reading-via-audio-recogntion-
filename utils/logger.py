# utils/logger.py
import time

class Logger:
    @staticmethod
    def log(component, msg, level="INFO"):
        ts = time.strftime("%H:%M:%S")
        prefix = {"INFO": "ℹ️", "WARN": "⚠️", "ERR": "❌", "OK": "✅"}.get(level, "·")
        print(f"[{ts}] {prefix} [{component}] {msg}")