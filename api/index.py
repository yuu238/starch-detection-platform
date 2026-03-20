"""
Vercel Serverless 入口 - 智能淀粉检测系统
将 FastAPI app 暴露给 Vercel Python Runtime
"""
import sys
import os

# 将项目根目录加入 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.main import app  # noqa: F401 - Vercel 需要名为 app 的 ASGI 对象
