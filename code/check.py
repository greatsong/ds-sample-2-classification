# 검증 스크립트: main.py와 같은 계산을 터미널에서 돌려 README·log·slides에 적은 모든 수치를 출력한다.
# 실행: 저장소 맨 위 폴더에서  python3 code/check.py
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

DATA = Path(__file__).resolve().parent.parent / "data" / "eafc25_players.csv"
FEATURES = ["pace", "shooting", "passing", "dribbling", "defending", "physic"]
ORDER = ["DF", "MF", "FW"]

DF_POS = {"CB", "LB", "RB", "LWB", "RWB"}
MF_POS = {"CDM", "CM", "CAM", "LM", "RM"}


def group(pos, cdm_to_df=False):
    """포지션 약어 하나를 DF·MF·FW·GK 묶음으로 바꾼다."""
    if pos == "GK":
        return "GK"
    if cdm_to_df and pos == "CDM":
        return "DF"
    if pos in DF_POS:
        return "DF"
    if pos in MF_POS:
        return "MF"
    return "FW"


def run(df, label_col):
    X = df[FEATURES]
    y = df[label_col]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    majority = y_tr.value_counts().idxmax()
    base_acc = (y_te == majority).mean()
    logit = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
    tree = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X_tr, y_tr)
    pred = logit.predict(X_te)
    return {
        "n_train": len(X_tr), "n_test": len(X_te), "majority": majority, "base_acc": base_acc,
        "logit_acc": accuracy_score(y_te, pred),
        "tree_acc": accuracy_score(y_te, tree.predict(X_te)),
        "cm": confusion_matrix(y_te, pred, labels=ORDER),
        "prf": precision_recall_fscore_support(y_te, pred, labels=ORDER, zero_division=0),
        "test_counts": y_te.value_counts().reindex(ORDER),
        "wrong_pos": df.loc[y_te.index[pred != y_te.values], "first_pos"].value_counts(),
        "model": logit,
    }


def show(res, title):
    print(f"\n### {title}")
    print(f"학습 {res['n_train']}명 · 시험 {res['n_test']}명 · 시험 레이블 분포 {dict(res['test_counts'])}")
    print(f"기준선(다수 범주 {res['majority']}만 찍기) 정확도: {res['base_acc']:.3f}")
    print(f"로지스틱 회귀 정확도: {res['logit_acc']:.3f}")
    print(f"결정 트리(깊이 4) 정확도: {res['tree_acc']:.3f}")
    print(f"로지스틱이 기준선보다 높은 정도: {res['logit_acc'] - res['base_acc']:.3f}"
          f" (맞힌 인원 {round(res['base_acc'] * res['n_test'])}명 → {round(res['logit_acc'] * res['n_test'])}명)")
    print("혼동행렬(로지스틱, 행=실제, 열=예측, 순서 DF·MF·FW)")
    print(pd.DataFrame(res["cm"], index=["실제 " + c for c in ORDER], columns=["예측 " + c for c in ORDER]))
    p, r, f, s = res["prf"]
    print("범주별 지표(로지스틱)")
    for i, c in enumerate(ORDER):
        print(f"  {c}: 정밀도 {p[i]:.2f} · 재현율 {r[i]:.2f} · F1 {f[i]:.2f} · 시험 인원 {s[i]}")
    print(f"틀린 {int(res['wrong_pos'].sum())}명의 첫 포지션:", res["wrong_pos"].to_dict())


df = pd.read_csv(DATA)
print("## 데이터")
print(f"행 {len(df)} · 열 {df.shape[1]}")
print("여섯 능력치 결측:", int(df[FEATURES].isna().sum().sum()))
print("positions 결측:", int(df["positions"].isna().sum()))

pos_lists = df["positions"].str.split(",").apply(lambda xs: [x.strip() for x in xs])
df["first_pos"] = pos_lists.str[0]
df["label"] = df["first_pos"].apply(group)
df["label_cdm"] = df["first_pos"].apply(lambda p: group(p, cdm_to_df=True))

print("\n## 레이블")
print("첫 포지션이 GK인 선수:", int((df["first_pos"] == "GK").sum()))
print("positions 어디에든 GK가 있는 선수:", int(pos_lists.apply(lambda xs: "GK" in xs).sum()))
print("첫 포지션 분포:", df["first_pos"].value_counts().to_dict())
counts = df["label"].value_counts().reindex(ORDER)
print("레이블 분포:", counts.to_dict())
print("레이블 비율:", {k: round(v / len(df), 3) for k, v in counts.items()})

multi = pos_lists.apply(len) > 1
groups = pos_lists.apply(lambda xs: {group(x) for x in xs})
cross = groups.apply(len) > 1
print(f"포지션이 둘 이상 적힌 선수: {int(multi.sum())}명 ({multi.mean():.1%})")
print(f"적힌 포지션이 두 묶음 이상에 걸친 선수: {int(cross.sum())}명 ({cross.mean():.1%})")
print("묶음 조합별 인원:", groups[cross].apply(lambda g: "+".join(sorted(g))).value_counts().to_dict())
print("첫 포지션이 CDM인 선수:", int((df["first_pos"] == "CDM").sum()))

print("\n## 포지션별 능력치 평균")
print(df.groupby("label")[FEATURES].mean().reindex(ORDER).round(1))

main = run(df, "label")
show(main, "본 분석: 첫 포지션 기준 (CDM은 MF)")

print("\n## 슬라이더 예시(앱 입력 칸의 기본값 = 전체 평균을 반올림)")
avg = df[FEATURES].mean().round().astype(int)
print("전체 평균:", avg.to_dict())
proba = main["model"].predict_proba(pd.DataFrame([avg], columns=FEATURES))[0]
print("예측:", main["model"].predict(pd.DataFrame([avg], columns=FEATURES))[0],
      {c: round(float(p), 2) for c, p in zip(main["model"].classes_, proba)})

print("\n## 실험: CDM을 DF로 옮긴 경우")
print("레이블 분포:", df["label_cdm"].value_counts().reindex(ORDER).to_dict())
show(run(df, "label_cdm"), "CDM을 DF로 옮긴 레이블")

print("\n## 후보 모델 비교 (본 분석 레이블, 같은 분할: 시험 20%·층화·분할 난수 42, 모델 난수 0)")
# XGBoost는 로컬에서만 계산한다(stlite 앱에는 설치할 수 없어 앱은 이 출력값을 적어 보여 준다).
from xgboost import XGBClassifier

X, y = df[FEATURES], df["label"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
code_of = {c: i for i, c in enumerate(ORDER)}
candidates = {
    "로지스틱 회귀": LogisticRegression(max_iter=2000),
    "k-NN(k=5)": KNeighborsClassifier(n_neighbors=5),
    "결정 트리(깊이 4)": DecisionTreeClassifier(max_depth=4, random_state=0),
    "랜덤 포레스트(300그루)": RandomForestClassifier(n_estimators=300, random_state=0),
    "XGBoost(기본값)": XGBClassifier(random_state=0),
}
print("모델 | 정확도 | 맞힌 인원 | macro F1 | F1 DF·MF·FW | 재현율 DF·MF·FW")
for name, m in candidates.items():
    if name.startswith("XGBoost"):
        m.fit(X_tr, y_tr.map(code_of))
        pred = pd.Series(m.predict(X_te)).map(dict(enumerate(ORDER))).values
    else:
        m.fit(X_tr, y_tr)
        pred = m.predict(X_te)
    p, r, f, _ = precision_recall_fscore_support(y_te, pred, labels=ORDER, zero_division=0)
    print(f"{name} | {accuracy_score(y_te, pred):.3f} | {int((pred == y_te.values).sum())}명"
          f" | {f1_score(y_te, pred, average='macro'):.3f}"
          f" | {' · '.join(f'{v:.2f}' for v in f)} | {' · '.join(f'{v:.2f}' for v in r)}")

print("\n## 로지스틱 회귀 계수 (능력치 1점이 오를 때 그 포지션 점수가 바뀌는 양)")
coef = pd.DataFrame(main["model"].coef_, index=main["model"].classes_, columns=FEATURES).reindex(ORDER)
print(coef.round(3))

print("\n## 인사이트 (README '우리 팀이 찾아낸 것'의 근거)")
model = main["model"]
X_te_main = X.loc[y_te.index]
pred_main = model.predict(X_te_main)
te = df.loc[y_te.index].copy()
te["pred"] = pred_main

print("\n### 인사이트 1: 수비가 높다고 수비수가 아니다")
df_min = int(df.loc[df["label"] == "DF", "defending"].min())
high_def = df[df["defending"] >= df_min]
print(f"DF 274명의 수비 능력치 최솟값: {df_min} (DF 전원이 수비 {df_min} 이상)")
print(f"수비 {df_min} 이상인 선수: {len(high_def)}명 · 그중 DF {int((high_def['label'] == 'DF').sum())}명"
      f"({(high_def['label'] == 'DF').mean():.1%}) · MF {int((high_def['label'] == 'MF').sum())}명"
      f" · FW {int((high_def['label'] == 'FW').sum())}명")
print("예시:", df.loc[df["name"] == "Rodri", ["name", "positions", "label", "defending"]].to_dict("records"))
print("능력치 하나만 넣은 로지스틱 회귀(같은 분할)")
for col in FEATURES:
    one = LogisticRegression(max_iter=2000).fit(X_tr[[col]], y_tr)
    p1 = one.predict(X_te[[col]])
    r1 = precision_recall_fscore_support(y_te, p1, labels=ORDER, zero_division=0)[1]
    print(f"  {col}: 정확도 {accuracy_score(y_te, p1):.3f} · 재현율 DF {r1[0]:.2f} · MF {r1[1]:.2f} · FW {r1[2]:.2f}")
print("DF 계수(크기 순):", coef.loc["DF"].reindex(coef.loc["DF"].abs().sort_values(ascending=False).index).round(3).to_dict())

print("\n### 인사이트 2: 틀린 선수는 경계 포지션, 특히 공격형 풀백")
fine = {"풀백(LB·RB)": {"LB", "RB"}, "센터백(CB)": {"CB"}, "CDM": {"CDM"},
        "윙어(LW·RW·LM·RM)": {"LW", "RW", "LM", "RM"}, "CM·CAM": {"CM", "CAM"}, "ST": {"ST"}}
for name, s in fine.items():
    t = te[te["first_pos"].isin(s)]
    n_wrong = int((t["pred"] != t["label"]).sum())
    print(f"  시험 {name}: {len(t)}명 중 {n_wrong}명 틀림({n_wrong / len(t):.0%})")
edge = {"LB", "RB", "LW", "RW", "LM", "RM", "CDM"}
wrong = te[te["pred"] != te["label"]]
print(f"틀린 {len(wrong)}명 중 풀백·윙어·CDM: {int(wrong['first_pos'].isin(edge).sum())}명")
print("틀린 풀백:")
print(wrong[wrong["first_pos"].isin({"LB", "RB"})][["name", "positions", "pred"] + FEATURES].to_string(index=False))
centers = df.groupby("label")[FEATURES].mean()
df["fine"] = df["first_pos"].map(lambda p: "FB" if p in {"LB", "RB"} else p)
fine_means = df.groupby("fine")[FEATURES].mean()
print("세부 포지션 평균에서 DF·MF·FW 평균까지의 거리(여섯 능력치 차이의 제곱합의 제곱근)")
for g in ["CB", "FB", "CDM", "CM"]:
    d = {c: round(float(((fine_means.loc[g] - centers.loc[c]) ** 2).sum() ** 0.5), 1) for c in ORDER}
    print(f"  {g}: {d}")

print("\n### 인사이트 3: MF는 가장 평평하고, 헷갈리는 선수가 MF로 모인다")
print("포지션별 평균 능력치 여섯 개의 표준편차:",
      centers.std(axis=1, ddof=0).reindex(ORDER).round(1).to_dict())
into_mf = te[(te["pred"] == "MF") & (te["label"] != "MF")]
print(f"MF로 잘못 답한 선수: {len(into_mf)}명 (틀린 {len(wrong)}명 중) · DF {int((into_mf['label'] == 'DF').sum())}명"
      f" · FW {int((into_mf['label'] == 'FW').sum())}명")
print(f"실제 MF {int((te['label'] == 'MF').sum())}명 중 맞힘 {int(((te['label'] == 'MF') & (te['pred'] == 'MF')).sum())}명")
print("MF로 잘못 답한 FW:")
print(into_mf[into_mf["label"] == "FW"][["name", "positions"] + FEATURES].to_string(index=False))
