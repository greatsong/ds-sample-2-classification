# 능력치 여섯 개로 축구 선수의 포지션을 맞힐 수 있을까 (FC25 선수 800명)
# 실행: 저장소 맨 위 폴더에서  streamlit run code/main.py
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

DATA = "data/eafc25_players.csv"
FEATURES = ["pace", "shooting", "passing", "dribbling", "defending", "physic"]
KOR = {"pace": "속력", "shooting": "슈팅", "passing": "패스",
       "dribbling": "드리블", "defending": "수비", "physic": "피지컬"}
ORDER = ["DF", "MF", "FW"]
NAME = {"DF": "수비수(DF)", "MF": "미드필더(MF)", "FW": "공격수(FW)"}
COLOR = {"DF": "#2b7fd6", "MF": "#2e9e6b", "FW": "#e0663a"}

# 후보 모델 비교: code/check.py를 로컬에서 실행해 얻은 값(XGBoost는 브라우저 앱에 설치할 수 없어 표 전체를 적어 둔다)
CANDIDATES = pd.DataFrame({
    "모델": ["기준선(MF만 찍기)", "로지스틱 회귀(최종)", "k-NN(k=5)", "결정 트리(깊이 4)", "랜덤 포레스트(300그루)", "XGBoost(기본값)"],
    "정확도": [0.450, 0.844, 0.856, 0.781, 0.850, 0.819],
    "맞힌 인원": ["72명", "135명", "137명", "125명", "136명", "131명"],
    "macro F1": ["", "0.845", "0.846", "0.778", "0.849", "0.813"],
    "F1 (DF · MF · FW)": ["", "0.85 · 0.84 · 0.84", "0.92 · 0.85 · 0.77", "0.81 · 0.78 · 0.75", "0.87 · 0.84 · 0.84", "0.88 · 0.81 · 0.76"],
    "재현율 (DF · MF · FW)": ["", "0.80 · 0.89 · 0.82", "0.89 · 0.88 · 0.76", "0.73 · 0.85 · 0.73", "0.82 · 0.90 · 0.79", "0.84 · 0.83 · 0.76"],
})

DF_POS = {"CB", "LB", "RB", "LWB", "RWB"}
MF_POS = {"CDM", "CM", "CAM", "LM", "RM"}


def group(pos):
    """첫 번째 포지션 약어를 DF·MF·FW·GK 묶음으로 바꾼다."""
    if pos == "GK":
        return "GK"
    if pos in DF_POS:
        return "DF"
    if pos in MF_POS:
        return "MF"
    return "FW"


@st.cache_data
def load():
    df = pd.read_csv(DATA)
    df["first_pos"] = df["positions"].str.split(",").str[0].str.strip()
    df["label"] = df["first_pos"].apply(group)
    df["multi"] = df["positions"].str.contains(",")
    return df


@st.cache_resource
def train(df):
    X = df[FEATURES]
    y = df["label"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    majority = y_tr.value_counts().idxmax()
    logit = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
    pred = logit.predict(X_te)
    return {
        "majority": majority,
        "base_acc": (y_te == majority).mean(),
        "logit_acc": accuracy_score(y_te, pred),
        "cm": confusion_matrix(y_te, pred, labels=ORDER),
        "prf": precision_recall_fscore_support(y_te, pred, labels=ORDER, zero_division=0),
        "model": logit,
        "n_test": len(y_te),
    }


st.set_page_config(page_title="능력치로 포지션 맞히기", page_icon="⚽", layout="wide")
df = load()
res = train(df)

st.title("⚽ 능력치 여섯 개로 포지션을 맞힐 수 있을까")
st.caption("FC25 선수 800명 · 분류(로지스틱 회귀, 후보 모델 다섯 가지 비교) · 당곡고 데이터 과학 팀 프로젝트 샘플")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["① 질문", "② 데이터 보기", "③ 기준선", "④ 모델 결과", "⑤ 해석·한계"])

with tab1:
    st.subheader("우리 팀의 질문")
    st.markdown("> **능력치 여섯 개(속력·슈팅·패스·드리블·수비·피지컬)만 보고 선수의 포지션(공격·미드필더·수비)을 맞힐 수 있는가**")
    st.write("게임에서 선수를 고를 때 능력치 모양만 봐도 포지션이 짐작됩니다. 그 감이 데이터로도 확인되는지 알아보았습니다.")
    st.write("종합 능력치(overall)·몸값·나이는 넣지 않았습니다. 포지션과 직접 관계가 없고, 질문이 '능력치 여섯 개만'이기 때문입니다.")

with tab2:
    st.subheader("레이블 만들기")
    st.write("positions 열에는 'RM,RW'처럼 여러 포지션이 쉼표로 적혀 있습니다. 첫 번째 포지션만 사용해 세 묶음으로 나눴습니다.")
    st.table(pd.DataFrame({
        "묶음": ["DF", "MF", "FW"],
        "포함한 포지션": ["CB · LB · RB · LWB · RWB", "CDM · CM · CAM · LM · RM", "나머지(이 데이터에서는 ST · LW · RW)"],
    }).set_index("묶음"))
    c1, c2 = st.columns(2)
    with c1:
        counts = df["label"].value_counts().reindex(ORDER).reset_index()
        counts.columns = ["포지션", "선수 수"]
        fig = px.bar(counts, x="포지션", y="선수 수", text="선수 수", color="포지션",
                     color_discrete_map=COLOR, title="레이블 분포")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.metric("포지션이 둘 이상 적힌 선수", f"{int(df['multi'].sum())}명", f"전체의 {df['multi'].mean():.1%}", delta_color="off")
        st.metric("골키퍼(GK)", "0명", "이 800명에는 골키퍼가 없음", delta_color="off")

    st.subheader("포지션별 능력치 평균")
    means = df.groupby("label")[FEATURES].mean().reindex(ORDER).rename(columns=KOR).reset_index()
    long = means.melt(id_vars="label", var_name="능력치", value_name="평균")
    fig = px.bar(long, x="능력치", y="평균", color="label", barmode="group",
                 color_discrete_map=COLOR, labels={"label": "포지션"}, text_auto=".1f")
    fig.update_yaxes(range=[0, 100])
    st.plotly_chart(fig, width="stretch")
    st.write("수비 능력치에서 세 묶음의 차이가 가장 큽니다. 슈팅은 수비수가 낮고, 패스는 미드필더가 가장 높습니다.")
    with st.expander("원본 데이터 일부 보기"):
        st.dataframe(df[["name", "positions", "label"] + FEATURES].head(30), width="stretch")

with tab3:
    st.subheader("기준선: 가장 많은 포지션만 찍기")
    st.write(f"학습 데이터에서 가장 많은 포지션은 {res['majority']}입니다. 시험 데이터 {res['n_test']}명 모두를 {res['majority']}라고 답하면 맞히는 비율이 기준선입니다.")
    st.metric("기준선 정확도", f"{res['base_acc']:.3f}")
    st.write("모델이 이 값보다 높아야 능력치에서 무언가를 배웠다고 말할 수 있습니다.")

with tab4:
    st.subheader("후보 모델 비교")
    st.write(f"이 앱에서 바로 계산한 값: 기준선 {res['base_acc']:.3f} · 로지스틱 회귀 {res['logit_acc']:.3f}")
    st.table(CANDIDATES.assign(정확도=CANDIDATES["정확도"].map("{:.3f}".format)).set_index("모델"))
    st.caption("표는 같은 학습 640명·시험 160명으로 code/check.py를 로컬에서 실행해 계산한 값입니다. XGBoost는 브라우저 앱에서 실행할 수 없습니다.")
    fig = px.bar(CANDIDATES, x="정확도", y="모델", orientation="h", text_auto=".3f", color="모델",
                 color_discrete_sequence=["#9aa3b5", "#2b7fd6", "#c7cedb", "#c7cedb", "#c7cedb", "#c7cedb"])
    fig.update_layout(showlegend=False, yaxis={"autorange": "reversed"})
    fig.update_xaxes(range=[0, 1])
    st.plotly_chart(fig, width="stretch")
    st.markdown(
        "**로지스틱 회귀를 고른 이유**\n"
        "- k-NN(0.856)과 랜덤 포레스트(0.850)가 정확도는 더 높지만, 시험 160명 가운데 2명과 1명 차이입니다.\n"
        "- 계수로 어느 능력치가 어느 포지션을 가르는지 읽을 수 있습니다. 수업 11차시에서 배운 모델입니다.\n"
        "- F1이 세 포지션 모두 0.84~0.85로 고르고, FW 재현율 0.82는 다섯 모델 가운데 가장 높습니다."
    )
    coef = pd.DataFrame(res["model"].coef_, index=res["model"].classes_, columns=[KOR[f] for f in FEATURES]).reindex(ORDER)
    st.subheader("로지스틱 회귀 계수")
    st.write("능력치 1점이 오를 때 그 포지션 점수가 바뀌는 양입니다. 양수면 그 포지션일 가능성을 높이고, 음수면 낮춥니다.")
    st.dataframe(coef.round(3), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("혼동행렬(로지스틱 회귀)")
        cm = pd.DataFrame(res["cm"], index=["실제 " + c for c in ORDER], columns=["예측 " + c for c in ORDER])
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", aspect="auto")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.subheader("범주별 정밀도·재현율·F1")
        p, r, f, s = res["prf"]
        table = pd.DataFrame({
            "포지션": [NAME[c] for c in ORDER],
            "정밀도": [f"{v:.2f}" for v in p],
            "재현율": [f"{v:.2f}" for v in r],
            "F1": [f"{v:.2f}" for v in f],
            "시험 인원": s,
        })
        st.table(table.set_index("포지션"))
        st.write("정밀도: 모델이 그 포지션이라고 답한 선수 가운데 실제로 맞은 비율")
        st.write("재현율: 실제 그 포지션인 선수 가운데 모델이 찾아낸 비율")
        st.write("F1: 정밀도와 재현율을 함께 반영한 값(조화평균)")

    st.subheader("직접 넣어 보기")
    st.write("능력치 여섯 개를 슬라이더로 정하면 모델이 포지션을 예측합니다. 처음 값은 800명의 평균입니다.")
    avg = df[FEATURES].mean().round().astype(int)
    cols = st.columns(3)
    values = {}
    for i, feat in enumerate(FEATURES):
        with cols[i % 3]:
            values[feat] = st.slider(KOR[feat], 20, 99, int(avg[feat]))
    x = pd.DataFrame([values], columns=FEATURES)
    model = res["model"]
    guess = model.predict(x)[0]
    proba = pd.DataFrame({"포지션": [NAME[c] for c in model.classes_], "확률": model.predict_proba(x)[0]})
    st.success(f"예측: **{NAME[guess]}**")
    fig = px.bar(proba, x="확률", y="포지션", orientation="h", text_auto=".2f")
    fig.update_xaxes(range=[0, 1])
    st.plotly_chart(fig, width="stretch")

with tab5:
    st.subheader("해석")
    st.write(f"기준선 {res['base_acc']:.3f}에서 로지스틱 회귀 {res['logit_acc']:.3f}로 올랐습니다. 능력치 모양만으로도 포지션을 꽤 잘 맞힙니다.")
    st.write("혼동행렬을 보면 FW와 DF는 서로 한 명도 섞이지 않았습니다. 틀린 선수는 모두 MF와의 경계에 있었고, 풀백(RB·LB)·윙어(LW·RW·RM)·수비형 미드필더(CDM)가 많았습니다.")
    st.write("수비형 미드필더를 DF로 옮겨 다시 돌리면 정확도는 0.856이 되지만, 공격수의 재현율이 0.82에서 0.64로 떨어집니다. 묶는 기준을 바꾸면 결과도 달라집니다.")
    st.subheader("한계")
    st.markdown(
        "- 상위 800명만 있는 표본입니다. 골키퍼는 한 명도 없습니다.\n"
        "- 포지션 레이블은 게임 제작사가 정한 분류입니다.\n"
        "- 800명 가운데 533명(66.6%)은 포지션이 둘 이상 적혀 있는데, 첫 번째 하나로 줄였습니다.\n"
        "- **이 모델로 실제 선수의 포지션을 정하면 안 됩니다.** 게임 능력치를 흉내 낸 결과일 뿐입니다."
    )
