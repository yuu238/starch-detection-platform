"""
智能淀粉检测系统 - FastAPI 后端
集成 hiapi.online AI 大模型（OpenAI 兼容格式）
"""
import io
import base64
import os
import json
from datetime import datetime
from typing import List, Optional
from urllib.parse import quote

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression
from openai import OpenAI

# ─────────────────────────────────────────────
# AI 客户端配置（hiapi.online）
# ─────────────────────────────────────────────
AI_CLIENT = OpenAI(
    api_key="sk-9tbyKDHE8nvrN44O41WMV8f9v37iRHxVm8T6bZbTY3lKJUu2",
    base_url="https://hiapi.online/v1"
)
AI_MODEL = "gemini-2.5-flash"   # 性价比最高，支持流式输出

# ─────────────────────────────────────────────
# 字体配置（支持中文）
# ─────────────────────────────────────────────
def setup_chinese_font():
    font_paths = [
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            prop = fm.FontProperties(fname=fp)
            plt.rcParams['font.family'] = prop.get_name()
            plt.rcParams['axes.unicode_minus'] = False
            return prop
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    return None

FONT_PROP = setup_chinese_font()

# ─────────────────────────────────────────────
# FastAPI 应用
# ─────────────────────────────────────────────
app = FastAPI(title="智能淀粉检测系统", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR    = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'static')
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'templates')

# 静态文件目录（本地运行时挂载，Vercel 环境下目录可能不存在则跳过）
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ─────────────────────────────────────────────
# 数据模型
# ─────────────────────────────────────────────
class StandardPoint(BaseModel):
    concentration: float
    absorbance: float

class AnalysisRequest(BaseModel):
    standard_points: List[StandardPoint]
    sample_absorbance: Optional[float] = None
    theory_concentration: Optional[float] = None

class AIAnalysisRequest(BaseModel):
    standard_points: List[StandardPoint]
    sample_absorbance: Optional[float] = None
    theory_concentration: Optional[float] = None
    calc_concentration: Optional[float] = None
    deviation: Optional[float] = None
    judgment: Optional[str] = None
    judgment_level: Optional[str] = None
    k: float
    b: float
    r_squared: float
    equation: str
    model: Optional[str] = "gemini-2.5-flash"

class AnalysisResult(BaseModel):
    k: float
    b: float
    r_squared: float
    equation: str
    chart_base64: str
    calc_concentration: Optional[float] = None
    deviation: Optional[float] = None
    judgment: Optional[str] = None
    judgment_level: Optional[str] = None

# ─────────────────────────────────────────────
# 核心计算
# ─────────────────────────────────────────────
def run_linear_regression(points: List[StandardPoint]):
    concentrations = np.array([p.concentration for p in points]).reshape(-1, 1)
    absorbances    = np.array([p.absorbance    for p in points])
    model = LinearRegression()
    model.fit(concentrations, absorbances)
    k  = float(model.coef_[0])
    b  = float(model.intercept_)
    r2 = float(model.score(concentrations, absorbances))
    return model, k, b, r2, concentrations, absorbances

def generate_chart(k, b, r2, concentrations, absorbances, font_prop=None) -> str:
    fig, ax = plt.subplots(figsize=(7, 5), dpi=110)
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')

    ax.scatter(concentrations, absorbances, color='#f87171', s=80, zorder=5,
               label='标准品数据点' if font_prop else 'Standard Points')

    x_line = np.linspace(concentrations.min(), concentrations.max(), 200)
    y_line = k * x_line + b
    sign   = '+' if b >= 0 else '-'
    eq_lbl = f'A = {k:.4f}·C {sign} {abs(b):.4f}  (R²={r2:.4f})'
    ax.plot(x_line, y_line, color='#38bdf8', linewidth=2.5, label=eq_lbl)

    residuals = absorbances - (k * concentrations.flatten() + b)
    std_err   = np.std(residuals)
    ax.fill_between(x_line, y_line - std_err, y_line + std_err,
                    alpha=0.15, color='#38bdf8')

    ax.tick_params(colors='#94a3b8', labelsize=10)
    for spine in ['bottom','left']:
        ax.spines[spine].set_color('#334155')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fp_kw = {'fontproperties': font_prop} if font_prop else {}
    ax.set_xlabel('浓度 C (%)' if font_prop else 'Concentration C (%)',
                  color='#94a3b8', fontsize=11, **fp_kw)
    ax.set_ylabel('吸光度 A'   if font_prop else 'Absorbance A',
                  color='#94a3b8', fontsize=11, **fp_kw)
    ax.set_title('标准曲线拟合图' if font_prop else 'Standard Curve',
                 color='#e2e8f0', fontsize=13, fontweight='bold', **fp_kw)
    ax.legend(facecolor='#1e293b', edgecolor='#334155',
              labelcolor='#cbd5e1', fontsize=9)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def judge_sample(calc_c: float, theory_c: float):
    deviation = abs(calc_c - theory_c) / theory_c * 100 if theory_c != 0 else 0
    if deviation <= 5:
        return deviation, "正常 (Normal)", "normal"
    elif deviation <= 10:
        return deviation, "轻微异常 (Slight Anomaly)", "slight"
    else:
        return deviation, "疑似掺假 (Suspected Adulteration)", "danger"

# ─────────────────────────────────────────────
# PDF 生成
# ─────────────────────────────────────────────
def generate_pdf_report(result: dict, points: List[StandardPoint]) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle, Image, HRFlowable)
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    font_name = 'Helvetica'
    for fp, fn in [('/usr/share/fonts/truetype/wqy/wqy-microhei.ttc', 'WQYMicroHei'),
                   ('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',   'WQYZenHei')]:
        if os.path.exists(fp):
            try:
                pdfmetrics.registerFont(TTFont(fn, fp))
                font_name = fn
                break
            except Exception:
                pass

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                             leftMargin=2*cm, rightMargin=2*cm,
                             topMargin=2*cm, bottomMargin=2*cm)

    def ps(name, **kw):
        return ParagraphStyle(name, fontName=font_name, **kw)

    title_s   = ps('T', fontSize=20, textColor=colors.HexColor('#1e3a5f'),
                   spaceAfter=6, alignment=1, leading=28)
    sub_s     = ps('S', fontSize=11, textColor=colors.HexColor('#64748b'),
                   spaceAfter=4, alignment=1)
    time_s    = ps('Ti', fontSize=9, textColor=colors.HexColor('#9ca3af'), alignment=1)
    sec_s     = ps('Se', fontSize=13, textColor=colors.HexColor('#0f4c81'),
                   spaceBefore=14, spaceAfter=6, leading=18)
    body_s    = ps('B', fontSize=10, textColor=colors.HexColor('#374151'),
                   spaceAfter=4, leading=16)
    warn_s    = ps('W', fontSize=10, textColor=colors.HexColor('#b45309'))
    footer_s  = ps('F', fontSize=8, textColor=colors.HexColor('#9ca3af'), alignment=1)
    ai_s      = ps('AI', fontSize=10, textColor=colors.HexColor('#1e3a5f'),
                   spaceAfter=4, leading=16, backColor=colors.HexColor('#f0f9ff'))

    story = []
    story.append(Paragraph("智能淀粉检测系统", title_s))
    story.append(Paragraph("吸光度分析检测报告", sub_s))
    story.append(Paragraph(f"生成时间：{datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}", time_s))
    story.append(HRFlowable(width="100%", thickness=1.5,
                             color=colors.HexColor('#0f4c81'), spaceAfter=14))

    # 回归结果
    story.append(Paragraph("一、标准曲线拟合结果", sec_s))
    k, b, r2 = result['k'], result['b'], result['r_squared']
    sign = '+' if b >= 0 else '-'
    story.append(Paragraph(f"回归方程：A = {k:.4f} · C {sign} {abs(b):.4f}", body_s))
    story.append(Paragraph(f"相关系数 R² = {r2:.6f}", body_s))
    if r2 < 0.999:
        story.append(Paragraph("⚠ 警告：R² 低于 0.999，建议检查实验数据准确性。", warn_s))

    # 数据明细
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("二、标准品数据明细", sec_s))
    tdata = [['序号', '浓度 C (%)', '吸光度 A', '拟合值 Â', '残差']]
    for i, p in enumerate(points):
        fitted   = k * p.concentration + b
        residual = p.absorbance - fitted
        tdata.append([str(i+1), f"{p.concentration:.4f}", f"{p.absorbance:.4f}",
                      f"{fitted:.4f}", f"{residual:+.4f}"])
    tbl = Table(tdata, colWidths=[1.5*cm, 3.5*cm, 3.5*cm, 3.5*cm, 3.5*cm])
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e3a5f')),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('FONTNAME',   (0,0), (-1,-1), font_name),
        ('FONTSIZE',   (0,0), (-1,0), 10), ('FONTSIZE', (0,1), (-1,-1), 9),
        ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
        ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1),
         [colors.HexColor('#f8fafc'), colors.HexColor('#e2e8f0')]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING',    (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(tbl)

    # 曲线图
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("三、标准曲线图", sec_s))
    if result.get('chart_base64'):
        img_buf = io.BytesIO(base64.b64decode(result['chart_base64']))
        story.append(Image(img_buf, width=14*cm, height=9*cm))

    # 样品判定
    if result.get('calc_concentration') is not None:
        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph("四、样品智能判定结果", sec_s))
        sdata = [
            ['检测项目', '数值'],
            ['样品吸光度 A',    f"{result.get('sample_absorbance', '-'):.4f}"],
            ['理论预期浓度 (%)', f"{result.get('theory_concentration', '-'):.4f}"],
            ['实测计算浓度 (%)', f"{result['calc_concentration']:.4f}"],
            ['偏差值 (%)',       f"{result['deviation']:.2f}%"],
            ['判定结论',         result['judgment']],
        ]
        level = result.get('judgment_level', 'normal')
        jc = {'normal': '#065f46', 'slight': '#92400e', 'danger': '#7f1d1d'}.get(level, '#065f46')
        stbl = Table(sdata, colWidths=[7*cm, 8*cm])
        stbl.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e3a5f')),
            ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
            ('FONTNAME',   (0,0), (-1,-1), font_name),
            ('FONTSIZE',   (0,0), (-1,-1), 10),
            ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
            ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0,1), (-1,-2),
             [colors.HexColor('#f8fafc'), colors.HexColor('#e2e8f0')]),
            ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor(jc)),
            ('TEXTCOLOR',  (0,-1), (-1,-1), colors.white),
            ('FONTSIZE',   (0,-1), (-1,-1), 11),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
            ('TOPPADDING',    (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(stbl)

    # AI 分析报告
    if result.get('ai_analysis'):
        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph("五、AI 智能分析报告", sec_s))
        # 将 AI 文本按段落分割
        ai_text = result['ai_analysis']
        for para in ai_text.split('\n'):
            para = para.strip()
            if para:
                story.append(Paragraph(para.replace('**','').replace('*','').replace('#',''),
                                       ai_s))
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph(f"AI 模型：{result.get('ai_model', AI_MODEL)}  |  分析时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                                ps('AIM', fontSize=8, textColor=colors.HexColor('#9ca3af'))))

    # 页脚
    story.append(Spacer(1, 1*cm))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor('#cbd5e1'), spaceAfter=6))
    story.append(Paragraph("本报告由智能淀粉检测系统自动生成，仅供参考。", footer_s))

    doc.build(story)
    buf.seek(0)
    return buf.read()

# ─────────────────────────────────────────────
# 构建 AI 分析 Prompt
# ─────────────────────────────────────────────
def build_ai_prompt(req: AIAnalysisRequest) -> str:
    pts_text = "\n".join(
        f"  第{i+1}点：浓度 C = {p.concentration:.4f}%，吸光度 A = {p.absorbance:.4f}"
        for i, p in enumerate(req.standard_points)
    )
    judgment_map = {
        'normal': '✅ 正常',
        'slight': '⚠️ 轻微异常',
        'danger': '🚨 疑似掺假',
    }
    judgment_str = judgment_map.get(req.judgment_level, '未判定')

    sample_section = ""
    if req.calc_concentration is not None:
        sample_section = f"""
**样品检测结果：**
- 输入吸光度 A = {req.sample_absorbance:.4f}
- 理论预期浓度 = {req.theory_concentration:.4f}%
- 实测计算浓度 = {req.calc_concentration:.4f}%
- 相对偏差 = {req.deviation:.2f}%
- 初步判定 = {judgment_str}（{req.judgment}）
"""

    prompt = f"""你是一位专业的食品检测分析专家，擅长淀粉含量检测与掺假鉴别。
请根据以下检测数据，提供一份专业、详尽的中文分析报告。

## 检测数据

**标准曲线数据（共 {len(req.standard_points)} 个标准品）：**
{pts_text}

**线性回归结果：**
- 回归方程：{req.equation}
- 斜率 k = {req.k:.4f}（吸光度/浓度灵敏度）
- 截距 b = {req.b:.4f}（基线偏移）
- 相关系数 R² = {req.r_squared:.6f}
{sample_section}

## 请按以下结构输出分析报告（使用 Markdown 格式）：

### 1. 标准曲线质量评估
评估 R² 值、斜率合理性、数据点分布情况，判断本次实验的标准曲线质量等级（优秀/良好/需复测）。

### 2. 方法灵敏度分析
根据斜率 k 值分析检测方法的灵敏度，与典型淀粉检测方法（碘量法/酶法）的参考值进行比较。

### 3. 样品判定深度解读
{('详细解读实测浓度与理论值的偏差原因，分析可能的掺假物质或实验误差来源，给出置信度评估。' if req.calc_concentration is not None else '（本次未输入样品数据，请在录入样品吸光度后重新分析）')}

### 4. 实验质量建议
针对本次检测数据，提出 3-5 条具体的改进建议，包括实验操作、数据质量控制等方面。

### 5. 综合结论
用 2-3 句话给出本次检测的综合结论和可信度评级。

请保持专业、客观，使用中文输出，适当使用表格或列表增强可读性。"""
    return prompt

# ─────────────────────────────────────────────
# API 路由
# ─────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    # 支持多种路径，兼容本地和 Vercel 环境
    possible_paths = [
        os.path.join(TEMPLATES_DIR, 'index.html'),
        os.path.join(os.path.dirname(__file__), '..', 'frontend', 'templates', 'index.html'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'frontend', 'templates', 'index.html'),
    ]
    for html_path in possible_paths:
        if os.path.exists(html_path):
            with open(html_path, 'r', encoding='utf-8') as f:
                return HTMLResponse(content=f.read())
    raise HTTPException(status_code=404, detail="页面文件未找到")

@app.post("/api/analyze", response_model=AnalysisResult)
async def analyze(req: AnalysisRequest):
    if len(req.standard_points) < 2:
        raise HTTPException(status_code=400, detail="至少需要 2 个标准品数据点")

    _, k, b, r2, concentrations, absorbances = run_linear_regression(req.standard_points)
    chart_b64 = generate_chart(k, b, r2, concentrations, absorbances, FONT_PROP)
    sign = '+ ' if b >= 0 else '- '
    equation = f"A = {k:.4f}·C {sign}{abs(b):.4f}"

    result = AnalysisResult(k=k, b=b, r_squared=r2,
                             equation=equation, chart_base64=chart_b64)

    if req.sample_absorbance is not None and req.theory_concentration is not None and k != 0:
        calc_c = (req.sample_absorbance - b) / k
        deviation, judgment, level = judge_sample(calc_c, req.theory_concentration)
        result.calc_concentration = round(calc_c, 4)
        result.deviation          = round(deviation, 2)
        result.judgment           = judgment
        result.judgment_level     = level

    return result

@app.post("/api/ai-analyze")
async def ai_analyze_stream(req: AIAnalysisRequest):
    """流式 AI 分析接口（SSE 格式）"""
    prompt = build_ai_prompt(req)
    model  = req.model or AI_MODEL

    async def event_stream():
        try:
            # 发送开始信号
            yield f"data: {json.dumps({'type': 'start', 'model': model}, ensure_ascii=False)}\n\n"

            stream = AI_CLIENT.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system",
                     "content": "你是专业的食品检测分析专家，请用中文提供详尽的专业分析报告。"},
                    {"role": "user", "content": prompt}
                ],
                stream=True,
                temperature=0.7,
            )

            for chunk in stream:
                # 部分 API 会返回空 choices 的 chunk（如最终统计 chunk），需跳过
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    yield f"data: {json.dumps({'type': 'chunk', 'content': delta.content}, ensure_ascii=False)}\n\n"

            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
    )

@app.post("/api/export/pdf")
async def export_pdf(req: AIAnalysisRequest):
    """生成含 AI 分析的 PDF 报告"""
    if len(req.standard_points) < 2:
        raise HTTPException(status_code=400, detail="至少需要 2 个标准品数据点")

    _, k, b, r2, concentrations, absorbances = run_linear_regression(req.standard_points)
    chart_b64 = generate_chart(k, b, r2, concentrations, absorbances, FONT_PROP)

    result_dict: dict = {
        'k': k, 'b': b, 'r_squared': r2,
        'equation': req.equation,
        'chart_base64': chart_b64,
        'ai_analysis': None,
        'ai_model': req.model or AI_MODEL,
    }

    if req.sample_absorbance is not None and req.theory_concentration is not None and k != 0:
        calc_c = (req.sample_absorbance - b) / k
        deviation, judgment, level = judge_sample(calc_c, req.theory_concentration)
        result_dict.update({
            'sample_absorbance':   req.sample_absorbance,
            'theory_concentration': req.theory_concentration,
            'calc_concentration':  round(calc_c, 4),
            'deviation':           round(deviation, 2),
            'judgment':            judgment,
            'judgment_level':      level,
        })

    # 同步生成 AI 分析文本（PDF 需要完整文本）
    if req.calc_concentration is not None or req.k:
        try:
            prompt = build_ai_prompt(req)
            resp = AI_CLIENT.chat.completions.create(
                model=req.model or AI_MODEL,
                messages=[
                    {"role": "system",
                     "content": "你是专业的食品检测分析专家，请用中文提供详尽的专业分析报告。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.7,
            )
            result_dict['ai_analysis'] = resp.choices[0].message.content
        except Exception as e:
            result_dict['ai_analysis'] = f"AI 分析生成失败：{str(e)}"

    pdf_bytes = generate_pdf_report(result_dict, req.standard_points)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename_ascii = f"starch_report_{ts}.pdf"
    filename_utf8  = quote(f"淀粉检测报告_{ts}.pdf")

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition":
                 f"attachment; filename={filename_ascii}; filename*=UTF-8''{filename_utf8}"}
    )

@app.post("/api/export/csv")
async def export_csv(req: AnalysisRequest):
    rows = [{'浓度C(%)': p.concentration, '吸光度A': p.absorbance}
            for p in req.standard_points]
    df  = pd.DataFrame(rows)
    csv = df.to_csv(index=False, encoding='utf-8-sig')
    return StreamingResponse(
        io.BytesIO(csv.encode('utf-8-sig')),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=standard_data.csv"}
    )

@app.get("/api/models")
async def list_models():
    """返回可用模型列表"""
    return {
        "models": [
            {"id": "gemini-2.5-flash",        "name": "Gemini 2.5 Flash（推荐）",   "desc": "性价比最高，速度快"},
            {"id": "gemini-2.5-pro",           "name": "Gemini 2.5 Pro",             "desc": "100w 上下文，深度分析"},
            {"id": "gemini-2.5-flash-search",  "name": "Gemini 2.5 Flash + 搜索",    "desc": "支持实时搜索"},
            {"id": "gpt-4o",                   "name": "GPT-4o",                     "desc": "OpenAI 旗舰模型"},
            {"id": "gpt-5",                    "name": "GPT-5",                      "desc": "最新 GPT 模型"},
        ]
    }
