"""v4 포스터(poster.html)에 넣을 그래프 3개와 QR 코드 2개를 SVG로 만들어 끼워 넣는 스크립트.

v4 양식(2단 균등, 단 폭 260mm)에 맞춰 그래프 높이를 정했다. 혼동행렬(그림 2)을 크게 그린다.

저장소 맨 위 폴더에서 실행한다: python3 code/poster_figs.py
check.py와 같은 전처리·분할·모델을 사용하므로 그래프의 값은 check.py 수치와 같다.
포스터를 만들 때만 사용하는 도구라 matplotlib·segno·xgboost가 필요하다(앱 실행에는 필요 없음).
  pip install matplotlib segno xgboost

poster.html의 <!-- FIG:이름 --> 과 <!-- /FIG:이름 --> 사이를 새 SVG로 바꾼다.
  FIG:hist     2. 데이터  분포 그래프
  FIG:bars     4. 결과    평가지표 비교 묶음 막대(후보 모델 5종, 선택 모델만 강조 색, 기준선 정확도는 점선)
  FIG:result   4. 결과    결과 그래프(분류: 혼동행렬 히트맵)
  FIG:qr_repo  제목띠     저장소 QR
  FIG:qr_app   발치       앱 QR

[다른 샘플에서 이 파일을 쓸 때]
  · 공통 부분(설정·svg_of·qr_svg·inject)은 그대로 둔다.
  · "이 샘플의 데이터와 그래프" 부분만 바꾼다. 그래프 이름(FIG:이름) 다섯 개는 그대로 사용한다.
  · 그래프 폭은 COL_W(단 폭)로 고정한다. 폭을 맞춰야 그래프 글자 20pt가 포스터에서도 20pt로 보인다.
"""
import io
import logging
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # 없는 글꼴 경고 끄기
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np
import pandas as pd
import segno

# ───────────── 공통 설정(다섯 샘플이 같다) ─────────────
ROOT = Path(__file__).resolve().parent.parent
POSTER = ROOT / "poster.html"
ACCENT = "#1f3d7a"      # poster.html의 --accent와 같은 남색
GRAY = "#9aa0aa"        # 선택하지 않은 모델·나머지 점
DARK = "#4a4f57"        # 기준선
# 단 폭(인치) = (594mm - 좌우 여백 2×26mm - 단 사이 22mm) / 2 = 260mm. poster.html의 --pad-x·--gap을 바꾸면 다시 계산한다
COL_W = 260 / 25.4
# v4 그래프 높이(인치): 단 폭은 같고 높이만 정한다
HIST_H, BARS_H, RESULT_H = 2.5, 3.6, 5.0

plt.rcParams.update({
    "svg.fonttype": "none",          # 글자를 글자로 남겨 브라우저 글꼴로 그린다
    "font.family": ["Apple SD Gothic Neo", "Noto Sans KR", "sans-serif"],
    "font.size": 20,                 # 포스터의 표·그림 설명 20pt와 맞춘다
    "axes.edgecolor": "#000", "axes.labelcolor": "#000",
    "xtick.color": "#000", "ytick.color": "#000",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 1.2,
    "svg.hashsalt": "poster",        # 다시 만들어도 같은 SVG가 나오게 한다
})


def svg_of(fig):
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0.05, transparent=True)
    plt.close(fig)
    s = buf.getvalue()
    s = s[s.index("<svg"):]                       # XML 머리말·DOCTYPE 제거
    s = re.sub(r'\swidth="[^"]+"\sheight="[^"]+"', "", s, count=1)  # 크기는 CSS가 정한다
    s = re.sub(r"<metadata>.*?</metadata>", "", s, flags=re.S)
    return s


def qr_svg(url):
    qr = segno.make(url, error="m")
    n = qr.symbol_size(scale=1, border=2)[0]
    buf = io.BytesIO()
    qr.save(buf, kind="svg", xmldecl=False, svgns=True, scale=1, border=2, dark="#000")
    s = buf.getvalue().decode()
    s = re.sub(r'\swidth="\d+"\sheight="\d+"', f' viewBox="0 0 {n} {n}" shape-rendering="crispEdges"', s, count=1)
    return s.replace("<svg ", f'<svg role="img" aria-label="{url}" ', 1)


def inject(figs):
    html = POSTER.read_text(encoding="utf-8")
    for name, svg in figs.items():
        pat = re.compile(rf"(<!-- FIG:{name} -->).*?(<!-- /FIG:{name} -->)", re.S)
        if not pat.search(html):
            raise SystemExit(f"poster.html에 FIG:{name} 표시가 없습니다")
        html = pat.sub(lambda m: m.group(1) + svg + m.group(2), html)
    POSTER.write_text(html, encoding="utf-8")


def bar_chart(names, values, pick, base, fmt, xlabel, height=4.6):
    """평가지표 비교 가로 막대. pick=선택 모델 이름, base=기준선 이름."""
    fig, ax = plt.subplots(figsize=(COL_W, height))
    y = np.arange(len(names))[::-1]
    colors = [ACCENT if n == pick else DARK if n == base else GRAY for n in names]
    ax.barh(y, values, color=colors, height=0.66)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    for t, n in zip(ax.get_yticklabels(), names):
        if n == pick:
            t.set_color(ACCENT)
            t.set_fontweight("bold")
    vmax = max(values)
    for yi, v, n in zip(y, values, names):
        ax.text(v + vmax * 0.012, yi, fmt(v), va="center", fontsize=20,
                color=ACCENT if n == pick else "#000", fontweight="bold" if n == pick else "normal")
    ax.set_xlim(0, vmax * 1.16)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda t, _: f"{t:,.0f}" if t >= 1 or t == 0 else f"{t:.2f}"))
    ax.set_xlabel(xlabel)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return svg_of(fig)


# ───────────── 이 샘플의 데이터와 그래프(샘플마다 바꾼다) ─────────────
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

DATA = ROOT / "data" / "eafc25_players.csv"
REPO_URL = "https://github.com/greatsong/ds-sample-2-classification"
APP_URL = "https://greatsong.github.io/ds-sample-2-classification/"

FEATURES = ["pace", "shooting", "passing", "dribbling", "defending", "physic"]
FEAT_KO = ["속력", "슈팅", "패스", "드리블", "수비", "피지컬"]
ORDER = ["DF", "MF", "FW"]
DF_POS = {"CB", "LB", "RB", "LWB", "RWB"}
MF_POS = {"CDM", "CM", "CAM", "LM", "RM"}


def group(pos):
    """check.py의 group()과 같다(본 분석: CDM은 MF)."""
    if pos == "GK":
        return "GK"
    if pos in DF_POS:
        return "DF"
    if pos in MF_POS:
        return "MF"
    return "FW"


# check.py와 같은 데이터 준비·분할
df = pd.read_csv(DATA)
df["label"] = df["positions"].str.split(",").str[0].str.strip().apply(group)
X, y = df[FEATURES], df["label"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 그래프 1(FIG:hist): 포지션별 능력치 평균(800명) 묶음 막대
means = df.groupby("label")[FEATURES].mean().reindex(ORDER)
pos_color = {"DF": ACCENT, "MF": GRAY, "FW": DARK}
pos_name = {"DF": f"DF 수비수 {(y == 'DF').sum()}명", "MF": f"MF 미드필더 {(y == 'MF').sum()}명",
            "FW": f"FW 공격수 {(y == 'FW').sum()}명"}
fig, ax = plt.subplots(figsize=(COL_W, HIST_H))
xs = np.arange(len(FEATURES))
w = 0.27
for i, c in enumerate(ORDER):
    ax.bar(xs + (i - 1) * w, means.loc[c], width=w, color=pos_color[c], label=pos_name[c])
ax.set_xticks(xs)
ax.set_xticklabels(FEAT_KO)
ax.set_ylim(0, 100)
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_ylabel("능력치 평균")
ax.tick_params(axis="x", length=0)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3, frameon=False, fontsize=20,
          handlelength=1.0, handletextpad=0.4, columnspacing=1.2)
svg_hist = svg_of(fig)

# 그래프 2(FIG:bars): 후보 모델 다섯 개의 정확도·재현율(macro)·F1(macro)
# check.py "후보 모델 비교"와 같은 모델·분할. 기준선은 정확도만 뜻이 있어 막대 대신 점선으로 그린다
code_of = {c: i for i, c in enumerate(ORDER)}
majority = y_tr.value_counts().idxmax()
rows = [("MF만 찍기(기준선)", np.full(len(y_te), majority))]
models = [
    ("로지스틱 회귀(선택)", LogisticRegression(max_iter=2000)),
    ("k-NN(k=5)", KNeighborsClassifier(n_neighbors=5)),
    ("결정 트리(깊이 4)", DecisionTreeClassifier(max_depth=4, random_state=0)),
    ("랜덤 포레스트", RandomForestClassifier(n_estimators=300, random_state=0)),
    ("XGBoost", XGBClassifier(random_state=0)),
]
for name, m in models:
    if name == "XGBoost":
        m.fit(X_tr, y_tr.map(code_of))
        pred = pd.Series(m.predict(X_te)).map(dict(enumerate(ORDER))).values
    else:
        m.fit(X_tr, y_tr)
        pred = m.predict(X_te)
    rows.append((name, pred))
PICK, BASE = "로지스틱 회귀(선택)", "MF만 찍기(기준선)"
base_acc = accuracy_score(y_te, rows[0][1])
scores = []
for name, pred in rows[1:]:
    _, r, f, _ = precision_recall_fscore_support(y_te, pred, labels=ORDER, zero_division=0)
    scores.append((name, accuracy_score(y_te, pred), r.mean(), f.mean()))

METRICS = ["정확도", "재현율(macro)", "F1(macro)"]
ALPHA = [1.0, 0.62, 0.34]           # 지표는 같은 색의 진하기로 구분한다
fig, ax = plt.subplots(figsize=(COL_W * 0.6, BARS_H))   # 오른쪽 40%에는 값 세 열을 적는다
gh = 0.26                            # 막대 하나의 높이
centers = np.arange(len(scores))[::-1] * 1.0
VAL_X = [1.10, 1.36, 1.62]           # 값 열의 위치(막대 축 좌표 기준)
for gi, (name, *vals) in enumerate(scores):
    col = ACCENT if name == PICK else DARK if name == BASE else GRAY
    for k, v in enumerate(vals):
        ax.barh(centers[gi] + (1 - k) * gh, v, height=gh * 0.92, color=col, alpha=ALPHA[k])
        ax.text(VAL_X[k], centers[gi], f"{v:.3f}", va="center", ha="right", fontsize=20, clip_on=False,
                color=ACCENT if name == PICK else "#000", fontweight="bold" if name == PICK else "normal")
top = centers[0] + 0.75
for k, lab in enumerate(METRICS):
    ax.add_patch(plt.Rectangle((VAL_X[k] - 0.235, top + 0.05), 0.04, 0.3, color=GRAY, alpha=ALPHA[k], clip_on=False))
    ax.text(VAL_X[k] - 0.185, top + 0.2, lab.replace("(macro)", ""), va="center", ha="left", fontsize=20, clip_on=False)
ax.axvline(base_acc, color="#000", linestyle=(0, (4, 3)), linewidth=1.4)
ax.text(base_acc + 0.015, top + 0.2, f"기준선 {base_acc:.3f}", va="center", ha="left", fontsize=20)
ax.set_ylim(centers[-1] - 0.55, top + 0.45)
ax.set_yticks(centers)
ax.set_yticklabels([s[0] for s in scores])
for t in ax.get_yticklabels():
    if t.get_text() == PICK:
        t.set_color(ACCENT)
        t.set_fontweight("bold")
ax.set_xlim(0, 1.0)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.xaxis.set_major_formatter(FuncFormatter(lambda t, _: "0" if t == 0 else f"{t:g}"))
ax.set_xlabel("시험 160명 기준 값(클수록 좋음)")
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
svg_bars = svg_of(fig)

# 그래프 3(FIG:result): 로지스틱 회귀의 혼동행렬(시험 160명, 행=실제, 열=예측)
logit = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
cm = confusion_matrix(y_te, logit.predict(X_te), labels=ORDER)
cmap = LinearSegmentedColormap.from_list("accent", ["#ffffff", ACCENT])
fig, ax = plt.subplots(figsize=(COL_W, RESULT_H))
ax.imshow(cm, cmap=cmap, vmin=0, vmax=cm.max(), aspect="auto")
for i in range(3):
    for j in range(3):
        v = cm[i, j]
        ax.text(j, i, f"{v}명", ha="center", va="center", fontsize=34, fontweight="bold",
                color="#fff" if v > cm.max() * 0.5 else "#000")
ax.set_xticks(range(3))
ax.set_xticklabels([f"예측 {c}" for c in ORDER])
ax.set_yticks(range(3))
ax.set_yticklabels([f"실제 {c} {cm[i].sum()}명" for i, c in enumerate(ORDER)])
ax.xaxis.set_ticks_position("top")
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_xticks(np.arange(-0.5, 3), minor=True)
ax.set_yticks(np.arange(-0.5, 3), minor=True)
ax.grid(which="minor", color="#fff", linewidth=3)
ax.tick_params(which="minor", length=0)
svg_result = svg_of(fig)

inject({"hist": svg_hist, "bars": svg_bars, "result": svg_result,
        "qr_repo": qr_svg(REPO_URL), "qr_app": qr_svg(APP_URL)})
print(f"기준선 정확도 {base_acc:.3f}")
for name, acc, rec, f1 in scores:
    print(f"{name}: 정확도 {acc:.3f} · 재현율(macro) {rec:.3f} · F1(macro) {f1:.3f}")
print("혼동행렬(로지스틱):", cm.tolist())
print(f"poster.html 갱신: 그래프 3개, QR 2개 (시험 데이터 {len(y_te)}명, 맞힌 인원 {int(np.trace(cm))}명)")
