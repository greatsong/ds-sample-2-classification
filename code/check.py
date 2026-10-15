# 검증 스크립트: main.py와 같은 계산을 터미널에서 돌려 README·log·slides에 적은 모든 수치를 출력한다.
# 실행: 저장소 맨 위 폴더에서  python3 code/check.py
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

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
