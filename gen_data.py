"""Synthetic oil & gas centrifugal-pump sensor dataset + ML demo.
SYNTHETIC DATASET CREATED FOR ACADEMIC DEMONSTRATION — not real industrial measurements."""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

rng = np.random.default_rng(42)
N = 6000                      # 5-minute samples (~20.8 days)
t = pd.date_range("2026-01-01", periods=N, freq="5min")

# Four run-to-failure episodes; maintenance resets degradation after each failure
lengths = [1700, 1400, 1600, 1300]
deg = np.zeros(N)
episode = np.zeros(N, dtype=int)
steps_to_fail = np.zeros(N)
i = 0
for e, L in enumerate(lengths):
    onset = int(L * rng.uniform(0.45, 0.6))          # healthy period, then wear begins
    x = np.zeros(L)
    k = np.arange(L - onset)
    x[onset:] = (np.exp(k / (L - onset) * 3.2) - 1) / (np.exp(3.2) - 1)
    deg[i:i + L] = np.clip(x, 0, 1)
    episode[i:i + L] = e + 1
    steps_to_fail[i:i + L] = np.arange(L)[::-1]
    i += L

d = deg
noise = lambda s: rng.normal(0, s, N)
temperature = 70 + 26 * d + 2.0 * np.sin(np.arange(N) * 2 * np.pi / 288) + noise(1.2)   # °C, daily cycle
vibration = 2.6 + 6.5 * d ** 1.4 + noise(0.25)                                         # mm/s RMS
flow_rate = 112 - 26 * d + noise(1.8)                                                  # L/min
pressure = 106 - 11 * d + noise(1.0)                                                   # psi
rpm = 2950 - 60 * d + noise(8)
motor_current = 25 + 11 * d + noise(0.7)                                               # A
gas_concentration = np.clip(4 + 18 * d ** 2 + noise(1.5), 0, None)                     # ppm (seal leak proxy)
equipment_health = np.clip(100 * (1 - d) + noise(1.0), 0, 100)
failure_probability = 1 / (1 + np.exp(-(d - 0.62) * 12))
failure = (steps_to_fail <= 288).astype(int)    # label: failure within next 24 h

# inject a few sensor glitches + missing values for the cleaning step
gl = rng.choice(N, 12, replace=False); vibration[gl] += rng.uniform(5, 9, 12)
df = pd.DataFrame(dict(
    timestamp=t, temperature=temperature.round(2), pressure=pressure.round(2),
    flow_rate=flow_rate.round(2), vibration=vibration.round(3), rpm=rpm.round(0),
    motor_current=motor_current.round(2), gas_concentration=gas_concentration.round(2),
    equipment_health=equipment_health.round(1), failure=failure,
    failure_probability=failure_probability.round(4)))
for c in ["temperature", "pressure", "flow_rate"]:
    df.loc[rng.choice(N, 25, replace=False), c] = np.nan
df.to_csv("synthetic_oil_gas_pump_sensors.csv", index=False)

# ---------- ML demo ----------
X = df.copy()
X[["temperature", "pressure", "flow_rate"]] = X[["temperature", "pressure", "flow_rate"]].interpolate()
for c in ["temperature", "vibration", "flow_rate", "motor_current", "pressure"]:
    X[f"{c}_rm12"] = X[c].rolling(12, min_periods=1).mean()      # 1-hour rolling mean
    X[f"{c}_rs12"] = X[c].rolling(12, min_periods=1).std().fillna(0)
X["temp_rate"] = X["temperature"].diff(12).fillna(0)
X["vib_rms12"] = np.sqrt((X["vibration"] ** 2).rolling(12, min_periods=1).mean())
X["flow_var"] = X["flow_rate"].rolling(36, min_periods=1).std().fillna(0)
feats = [c for c in X.columns if c not in
         ("timestamp", "failure", "failure_probability", "equipment_health")]

train = episode <= 3          # time-based split: last episode held out
Xtr, Xte, ytr, yte = X.loc[train, feats], X.loc[~train, feats], X.loc[train, "failure"], X.loc[~train, "failure"]

rf = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=0).fit(Xtr, ytr)
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")).fit(Xtr, ytr)
res = {}
test_predictions = {}
for name, m in [("Random Forest", rf), ("Logistic Regression", lr)]:
    p = m.predict_proba(Xte)[:, 1]; yh = (p >= 0.5).astype(int)
    res[name] = dict(accuracy=accuracy_score(yte, yh), precision=precision_score(yte, yh),
                     recall=recall_score(yte, yh), f1=f1_score(yte, yh), roc_auc=roc_auc_score(yte, p))
    test_predictions[name] = dict(y_true=yte.astype(int).tolist(),
                                  y_score=p.tolist(), y_pred=yh.tolist())
imp = pd.Series(rf.feature_importances_, index=feats).sort_values(ascending=False)
iso = IsolationForest(contamination=0.05, random_state=0).fit(X.loc[train & (deg < 0.05), feats])
anom = iso.predict(X[feats]) == -1

corr = X[["temperature", "pressure", "flow_rate", "vibration", "motor_current",
          "gas_concentration", "equipment_health"]].corr()["equipment_health"].drop("equipment_health")

# chart series from the held-out episode, hourly
ep = X[~train].iloc[::12].reset_index(drop=True)
ep_indices = X.index[~train][::12]
out = dict(
    rows=len(df), failure_rate=float(df.failure.mean()),
    metrics=res,
    test_predictions=test_predictions,
    importance={k: float(v) for k, v in imp.head(8).items()},
    corr_health={k: float(v) for k, v in corr.items()},
    anomaly_rate=float(anom.mean()),
    series=dict(hour=list(range(len(ep))),
                timestamp=ep.timestamp.astype(str).tolist(),
                temperature=ep.temperature.round(1).tolist(),
                vibration=ep.vibration.round(2).tolist(),
                motor_current=ep.motor_current.round(2).tolist(),
                health=ep.equipment_health.round(1).tolist(),
                failure=ep.failure.astype(int).tolist(),
                anomaly=anom[ep_indices].astype(int).tolist(),
                fprob=(ep.failure_probability * 100).round(1).tolist()))
json.dump(out, open("demo_results.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ("series", "test_predictions")}, indent=1))
print("hours in held-out episode:", len(ep))
