# 🧪 智能淀粉吸光度分析平台

基于 **FastAPI + Tailwind CSS** 构建的专业食品检测 Web 平台，实现淀粉含量标准曲线拟合、AI 掺假判定与 PDF 报告生成。

## ✨ 功能特性

- **酷炫加载动画**：DNA 双螺旋粒子动效 + 扫描线 + 进度条
- **线性回归算法**：实现 `A = k·C + b`，输出斜率 k、截距 b、R² 相关系数
- **AI 智能判定**：偏差 ≤5% 正常 / 5~10% 轻微异常 / >10% 疑似掺假
- **AI 深度分析**：集成 hiapi.online，支持 Gemini 2.5 / GPT-4o / GPT-5 流式输出
- **PDF 报告生成**：自动生成含图表、数据表、AI 分析的专业检测报告
- **CSV 数据导出**：一键导出标准品原始数据

## 🚀 快速启动

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# 访问
open http://localhost:8000
```

## 🏗️ 技术架构

| 层级 | 技术栈 |
|------|--------|
| 后端 | FastAPI + scikit-learn + ReportLab + Matplotlib |
| 前端 | HTML5 + Tailwind CSS + 原生 JavaScript |
| AI   | OpenAI 兼容 API（hiapi.online）|
| 字体 | WQY 微米黑（中文支持）|

## 📊 算法说明

标准曲线采用最小二乘线性回归：

```
A = k · C + b
```

其中 A 为吸光度，C 为浓度（%），k 为斜率，b 为截距。

样品实测浓度：`C_calc = (A_sample - b) / k`

偏差计算：`deviation = |C_calc - C_theory| / C_theory × 100%`

## 🤖 AI 支持模型

- Gemini 2.5 Flash（推荐，最快）
- Gemini 2.5 Pro（深度分析）
- Gemini 2.5 Flash + 搜索
- GPT-4o
- GPT-5

## 📄 许可证

MIT License
