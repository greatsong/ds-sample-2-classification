# ③④ 구술 평가 설명용 샘플 앱: 모델 후보·베이스라인·평가지표 (FC25 선수 800명)
# 실행: 저장소 맨 위 폴더에서  streamlit run code/oral34.py
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except ImportError:  # 브라우저 앱(stlite)에는 xgboost를 설치할 수 없다
    XGBClassifier = None

DATA = "data/eafc25_players.csv"
FEATURES = ["pace", "shooting", "passing", "dribbling", "defending", "physic"]
KOR = {"pace": "속력", "shooting": "슈팅", "passing": "패스",
       "dribbling": "드리블", "defending": "수비", "physic": "피지컬"}
ORDER = ["DF", "MF", "FW"]
NAME = {"DF": "수비수(DF)", "MF": "미드필더(MF)", "FW": "공격수(FW)"}
COLOR = {"DF": "#2b7fd6", "MF": "#2e9e6b", "FW": "#e0663a"}
SPLIT_SEED, MODEL_SEED, TEST_SIZE = 42, 0, 0.2
FINAL = "로지스틱 회귀"

# xgboost가 없는 환경에서 보여 줄 값: code/check.py를 로컬에서 실행해 얻은 결과
XGB_FALLBACK = {"정확도": 0.819, "맞힌 인원": 131, "macro F1": 0.813,
                "DF 재현율": 0.84, "MF 재현율": 0.83, "FW 재현율": 0.76}

DF_POS = {"CB", "LB", "RB", "LWB", "RWB"}
MF_POS = {"CDM", "CM", "CAM", "LM", "RM"}


def group(pos):
    """첫 번째 포지션 약어를 DF·MF·FW 묶음으로 바꾼다(이 데이터에는 GK가 없다)."""
    if pos in DF_POS:
        return "DF"
    if pos in MF_POS:
        return "MF"
    return "FW"


def ask(*questions):
    """이 탭이 답하는 선생님 질문을 맨 위에 보여 준다."""
    st.info("**이 화면으로 답하는 선생님 질문**\n" + "\n".join(f"- {q}" for q in questions))


@st.cache_data
def load():
    df = pd.read_csv(DATA)
    df["first_pos"] = df["positions"].str.split(",").str[0].str.strip()
    df["label"] = df["first_pos"].apply(group)
    return df


@st.cache_resource
def train_all(df):
    X, y = df[FEATURES], df["label"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=SPLIT_SEED, stratify=y)
    majority = y_tr.value_counts().idxmax()
    models = {
        "로지스틱 회귀": LogisticRegression(max_iter=2000),
        "k-NN(k=5)": KNeighborsClassifier(n_neighbors=5),
        "결정 트리(깊이 4)": DecisionTreeClassifier(max_depth=4, random_state=MODEL_SEED),
        "랜덤 포레스트(300그루)": RandomForestClassifier(n_estimators=300, random_state=MODEL_SEED),
    }
    if XGBClassifier is not None:
        models["XGBoost(기본값)"] = XGBClassifier(random_state=MODEL_SEED)

    rows, preds = [], {}
    base_pred = np.full(len(y_te), majority)
    for name, m in [("베이스라인(MF만 찍기)", None)] + list(models.items()):
        if m is None:
            pred = base_pred
        elif name.startswith("XGBoost"):
            code_of = {c: i for i, c in enumerate(ORDER)}
            m.fit(X_tr, y_tr.map(code_of))
            pred = pd.Series(m.predict(X_te)).map(dict(enumerate(ORDER))).values
        else:
            m.fit(X_tr, y_tr)
            pred = m.predict(X_te)
        preds[name] = pred
        _, r, _, _ = precision_recall_fscore_support(y_te, pred, labels=ORDER, zero_division=0)
        rows.append({"모델": name, "정확도": accuracy_score(y_te, pred), "맞힌 인원": int((pred == y_te.values).sum()),
                     "macro F1": f1_score(y_te, pred, labels=ORDER, average="macro", zero_division=0),
                     "DF 재현율": r[0], "MF 재현율": r[1], "FW 재현율": r[2], "계산": "이 앱"})
    if XGBClassifier is None:
        rows.append({"모델": "XGBoost(기본값)", **XGB_FALLBACK, "계산": "check.py 결과"})
    return {"table": pd.DataFrame(rows), "preds": preds, "models": models, "majority": majority,
            "X_te": X_te, "y_te": y_te, "n_tr": len(y_tr), "n_te": len(y_te)}


st.set_page_config(page_title="구술 ③④ 샘플 앱", page_icon="⚽", layout="wide")
df = load()
res = train_all(df)
table, y_te = res["table"], res["y_te"]
final_pred = res["preds"][FINAL]
base = table.iloc[0]
final = table[table["모델"] == FINAL].iloc[0]

st.title("⚽ 능력치 여섯 개로 포지션을 맞힐 수 있을까")
st.caption("팀 FC25(선생님이 만든 샘플 팀) · 구술 ③ 모델 후보·베이스라인 + ④ 평가지표·결과를 설명하는 화면")

PAGES = ["① 방법 고르기", "② 베이스라인", "③ 후보 모델 비교", "④ 고른 모델", "⑤ 결과 읽기"]
# 탭(st.tabs)은 화면 안의 버튼을 누를 때 첫 탭으로 돌아가므로, 고른 화면이 유지되는 가로 선택 버튼을 쓴다
page = st.radio("화면", PAGES, horizontal=True, label_visibility="collapsed", key="page")
st.divider()

if page == PAGES[0]:
    ask("질문 1. 회귀·분류·군집 가운데 무엇으로 풀었는지, 어떤 모델 후보를 비교했는지, 베이스라인은 무엇인지 설명해 주세요.")
    st.markdown("> **능력치 여섯 개(속력·슈팅·패스·드리블·수비·피지컬)만 보고 선수의 포지션을 맞힐 수 있는가**")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.table(pd.DataFrame({
            "확인할 것": ["답의 형태", "정답이 데이터에 있는가", "그래서 고른 방법"],
            "우리 팀": ["수비수·미드필더·공격수라는 범주", "있음(positions 열의 첫 번째 포지션)", "분류"],
        }).set_index("확인할 것"))
        st.write("답이 숫자였다면 회귀, 정답 없이 비슷한 선수끼리 묶으려 했다면 군집을 골랐을 것입니다.")
        st.write(f"후보 모델: {', '.join(table['모델'][1:])}")
        st.write(f"베이스라인: 모두 {res['majority']}로 찍기, 정확도 {base['정확도']:.3f} · 최종 모델: {FINAL}, 정확도 {final['정확도']:.3f}")
    with c2:
        counts = df["label"].value_counts().reindex(ORDER).reset_index()
        counts.columns = ["포지션", "선수 수"]
        fig = px.bar(counts, x="포지션", y="선수 수", text="선수 수", color="포지션",
                     color_discrete_map=COLOR, title="맞혀야 할 정답(포지션)의 분포")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")

if page == PAGES[1]:
    ask("질문 5. 베이스라인 모델은 무엇을 예측합니까?",
        "질문 9. 베이스라인보다 얼마나 나아졌습니까? 그 차이는 큰 편입니까?")
    st.write(f"베이스라인은 능력치를 보지 않고 시험 선수 {res['n_te']}명 모두를 학습 데이터에서 가장 많은 포지션인 "
             f"**{res['majority']}**로 찍습니다. 모델이 이 점수보다 높아야 능력치에서 무언가를 배웠다고 말할 수 있습니다.")
    c1, c2, c3 = st.columns(3)
    c1.metric("베이스라인", f"{base['맞힌 인원']}명 / {res['n_te']}명", f"정확도 {base['정확도']:.3f}", delta_color="off")
    c2.metric(FINAL, f"{final['맞힌 인원']}명 / {res['n_te']}명", f"정확도 {final['정확도']:.3f}", delta_color="off")
    c3.metric("더 맞힌 선수", f"{final['맞힌 인원'] - base['맞힌 인원']}명", f"정확도 +{final['정확도'] - base['정확도']:.3f}")
    fig = px.bar(table, x="정확도", y="모델", orientation="h", text_auto=".3f", color="모델",
                 color_discrete_map={m: ("#9aa3b5" if m.startswith("베이스라인") else "#2b7fd6" if m == FINAL else "#c7cedb")
                                     for m in table["모델"]},
                 title="베이스라인과 후보 모델의 정확도")
    fig.update_layout(showlegend=False, yaxis={"autorange": "reversed"})
    fig.update_xaxes(range=[0, 1])
    st.plotly_chart(fig, width="stretch")
    st.caption("회귀 팀은 학습 데이터 평균으로 모두 예측하는 평균 모델, 군집 팀은 묶음 개수 k를 정한 이유가 베이스라인 자리에 옵니다.")

if page == PAGES[2]:
    ask("질문 4. 비교한 후보 모델은 무엇이고, 같은 조건에서 비교했다는 근거는 무엇입니까?",
        "질문 6. 점수가 가장 높은 모델이 아닌 다른 모델을 고른다면 그 이유는 무엇입니까?",
        "질문 11. 다른 지표로 보면 결론이 달라집니까?")
    with st.container(border=True):
        st.markdown("**비교 조건(모든 모델이 같음)**")
        st.markdown(
            f"- 데이터 나누기: 학습 {res['n_tr']}명 · 시험 {res['n_te']}명({TEST_SIZE:.0%}), 세 포지션 비율을 똑같이 맞춤, 분할 난수 {SPLIT_SEED}\n"
            f"- 입력 변수: {' · '.join(KOR[f] for f in FEATURES)} 여섯 개\n"
            f"- 모델 난수: {MODEL_SEED} · 평가는 모두 같은 시험 {res['n_te']}명으로")
    metric = st.radio("어느 지표로 순서를 매길까요?", ["정확도", "macro F1", "DF 재현율", "MF 재현율", "FW 재현율"], horizontal=True)
    ranked = table.iloc[1:].sort_values(metric, ascending=False)
    top = ranked.iloc[0]["모델"]
    fig = px.bar(ranked, x=metric, y="모델", orientation="h", text_auto=".3f", color="모델",
                 color_discrete_map={m: ("#2b7fd6" if m == FINAL else "#c7cedb") for m in ranked["모델"]},
                 title=f"{metric} 기준 순서")
    fig.update_layout(showlegend=False, yaxis={"autorange": "reversed"})
    fig.update_xaxes(range=[0, 1])
    st.plotly_chart(fig, width="stretch")
    final_rank = list(ranked["모델"]).index(FINAL) + 1
    if top == FINAL:
        st.success(f"{metric} 기준으로는 {FINAL}가 1위입니다.")
    else:
        gap = ranked.iloc[0][metric] - final[metric]
        st.warning(f"{metric} 기준 1위는 {top}이고, {FINAL}는 {final_rank}위입니다(차이 {gap:.3f}).")
    shown = table.copy()
    for col in ["정확도", "macro F1", "DF 재현율", "MF 재현율", "FW 재현율"]:
        shown[col] = shown[col].map("{:.3f}".format)
    shown["맞힌 인원"] = shown["맞힌 인원"].map("{}명".format)
    st.dataframe(shown.set_index("모델"), width="stretch")
    if XGBClassifier is None:
        st.caption("XGBoost는 이 브라우저 환경에 설치할 수 없어 code/check.py를 로컬에서 실행한 값을 적었습니다.")
    best = table.iloc[1:].sort_values("정확도", ascending=False).iloc[0]
    st.markdown(
        f"**{FINAL}를 고른 이유**\n"
        f"- 정확도가 가장 높은 모델({best['모델']})과 비교해도 시험 {res['n_te']}명 가운데 {best['맞힌 인원'] - final['맞힌 인원']}명 차이라 순서가 쉽게 바뀔 수 있습니다.\n"
        "- FW 재현율로 순서를 바꿔 보면 로지스틱 회귀가 위로 올라옵니다. 인원이 적은 공격수를 덜 놓칩니다.\n"
        "- 계수로 어느 능력치가 어느 포지션을 가르는지 설명할 수 있습니다(④ 고른 모델 화면).")

if page == PAGES[3]:
    ask("질문 2. 선정한 모델은 어떤 특징이 있고, 어떤 경우에 많이 사용됩니까?",
        "질문 3. 왜 이 문제를 푸는 데 이 모델이 적합합니까?")
    st.markdown(
        "**로지스틱 회귀의 원리**: 능력치마다 계수를 곱해 더한 점수를 포지션마다 계산하고, "
        "그 점수를 확률로 바꿔 확률이 가장 높은 포지션으로 분류합니다.\n\n"
        "**많이 사용되는 경우**: 입력과 범주의 관계가 단순하고, 왜 그렇게 분류했는지 설명해야 할 때")
    model = res["models"][FINAL]
    coef = pd.DataFrame(model.coef_, index=model.classes_, columns=[KOR[f] for f in FEATURES]).reindex(ORDER)
    st.subheader("계수: 능력치 1점이 오를 때 그 포지션 점수가 바뀌는 양")
    fig = px.imshow(coef.round(3), text_auto=True, color_continuous_scale="RdBu", zmin=-0.3, zmax=0.3, aspect="auto",
                    labels={"x": "능력치", "y": "포지션", "color": "계수"})
    st.plotly_chart(fig, width="stretch")
    df_row = coef.loc["DF"]
    st.write(f"수비수 점수를 가장 크게 올리는 능력치는 {df_row.idxmax()}({df_row.max():+.3f}), "
             f"가장 크게 낮추는 능력치는 {df_row.idxmin()}({df_row.min():+.3f})입니다. "
             "여섯 능력치가 모두 0~100 척도라 계수의 크기를 비교할 수 있습니다.")

    st.subheader("한 선수로 계산 따라가 보기")
    who = st.selectbox("시험 데이터의 선수", df.loc[res["X_te"].index, "name"].tolist(), index=0)
    row = df[df["name"] == who].iloc[0]
    x = row[FEATURES].astype(float).values
    score = model.coef_ @ x + model.intercept_
    proba = np.exp(score - score.max()) / np.exp(score - score.max()).sum()
    steps = pd.DataFrame({"포지션": model.classes_, "점수(계수×능력치 합+절편)": score.round(2), "확률": proba.round(3)})
    steps = steps.set_index("포지션").reindex(ORDER)
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**{who}** · 실제 {row['first_pos']}({NAME[row['label']]})")
        st.table(pd.DataFrame({"능력치": [KOR[f] for f in FEATURES], "값": row[FEATURES].astype(int).values}).set_index("능력치").T)
        st.table(steps)
    with c2:
        fig = px.bar(steps.reset_index(), x="확률", y="포지션", orientation="h", text_auto=".2f",
                     color="포지션", color_discrete_map=COLOR, title="점수를 확률로 바꾼 결과")
        fig.update_layout(showlegend=False)
        fig.update_xaxes(range=[0, 1])
        st.plotly_chart(fig, width="stretch")
        guess = steps["확률"].idxmax()
        if guess == row["label"]:
            st.success(f"확률이 가장 높은 {NAME[guess]}로 분류합니다.")
        else:
            st.error(f"{NAME[guess]}로 분류했지만 실제는 {NAME[row['label']]}입니다.")
        if not np.allclose(proba, model.predict_proba(x.reshape(1, -1))[0]):
            st.caption("손으로 계산한 확률과 모델의 확률이 다릅니다.")

if page == PAGES[4]:
    ask("질문 7. 우리 모델의 결과를 평가지표와 베이스라인으로 설명하고, 그래프 하나를 골라 무엇을 뜻하는지 말해 주세요.",
        "질문 8. 우리 지표 값은 우리 데이터에서 무슨 뜻입니까?",
        "질문 10. 이 질문에서 왜 그 지표를 중요하게 봅니까?",
        "질문 12. 모델이 틀린 사례를 하나 보여 줄 수 있습니까?")
    cm = confusion_matrix(y_te, final_pred, labels=ORDER)
    p, r, f, s = precision_recall_fscore_support(y_te, final_pred, labels=ORDER, zero_division=0)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader(f"혼동행렬({FINAL})")
        cm_df = pd.DataFrame(cm, index=["실제 " + c for c in ORDER], columns=["예측 " + c for c in ORDER])
        fig = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues", aspect="auto")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")
        st.caption("가로줄은 실제 포지션, 세로줄은 모델의 답입니다. 대각선이 맞힌 선수입니다.")
    with c2:
        st.subheader("포지션별 지표")
        st.table(pd.DataFrame({"포지션": [NAME[c] for c in ORDER], "정밀도": [f"{v:.2f}" for v in p],
                               "재현율": [f"{v:.2f}" for v in r], "F1": [f"{v:.2f}" for v in f], "시험 인원": s})
                 .set_index("포지션"))
        st.write(f"정확도 {final['정확도']:.3f} · macro F1 {final['macro F1']:.3f}(세 포지션의 F1을 똑같이 평균한 값)")

    with st.container(border=True):
        st.markdown("**지표를 우리 말로**")
        pick = st.radio("포지션", ORDER, format_func=NAME.get, horizontal=True)
        i = ORDER.index(pick)
        said, real, hit = cm[:, i].sum(), cm[i, :].sum(), cm[i, i]
        st.markdown(
            f"- 정확도 {final['정확도']:.3f}: 시험 선수 100명이 있으면 {round(final['정확도'] * 100)}명쯤의 포지션을 맞힙니다.\n"
            f"- {NAME[pick]} 정밀도 {p[i]:.2f}: 모델이 {pick}라고 답한 {said}명 가운데 {hit}명이 실제 {pick}입니다.\n"
            f"- {NAME[pick]} 재현율 {r[i]:.2f}: 실제 {pick} {real}명 가운데 {hit}명을 찾아냈습니다. 놓친 {real - hit}명은 다른 포지션으로 답했습니다.")
        st.caption(f"미드필더가 가장 많아서(800명 가운데 {int((df['label'] == 'MF').sum())}명) 정확도만 보면 어느 포지션에서 틀렸는지 알 수 없습니다. 그래서 포지션별 재현율과 macro F1을 함께 봅니다.")

    st.subheader("틀린 선수 보기")
    wrong = df.loc[y_te.index].assign(예측=final_pred)
    wrong = wrong[wrong["label"] != wrong["예측"]]
    pairs = [f"실제 {a} → 예측 {b}" for a in ORDER for b in ORDER if a != b and ((wrong["label"] == a) & (wrong["예측"] == b)).any()]
    choice = st.selectbox(f"틀린 {len(wrong)}명 가운데 고르기", ["전체"] + pairs)
    if choice != "전체":
        a, b = choice.replace("실제 ", "").split(" → 예측 ")
        wrong = wrong[(wrong["label"] == a) & (wrong["예측"] == b)]
    st.dataframe(wrong[["name", "positions", "first_pos", "label", "예측"] + FEATURES]
                 .rename(columns={"name": "선수", "positions": "적힌 포지션", "first_pos": "첫 포지션", "label": "실제", **KOR}),
                 hide_index=True, width="stretch")
    st.caption("실제 DF를 FW로, 실제 FW를 DF로 답한 선수는 없습니다. 틀린 선수는 모두 MF와의 경계에서 나왔습니다.")
