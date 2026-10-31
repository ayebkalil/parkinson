"""Isolated row-preserving data protocol for model 1 and model 2.

No shared data is written. Source acquisition dates are unavailable in processed
data: this module certifies prefix/target alignment, not raw modality provenance.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import train_model2_lstm as historical

FEATURES = historical.FEATURE_COLS
STATIC_FEATURES = FEATURES[:-2]
TARGETS = ["target_np3tot_12m", "target_np3tot_24m", "prog_mcid_12m"]
DATA = Path("data/processed")


def write_json(path, value):
    def clean(v):
        if isinstance(v, dict):
            return {str(k): clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple, np.ndarray)):
            return [clean(x) for x in v]
        if isinstance(v, (np.integer, np.floating, np.bool_)):
            v = v.item()
        if isinstance(v, Path):
            return str(v)
        return None if isinstance(v, float) and not np.isfinite(v) else v
    Path(path).write_text(json.dumps(clean(value), indent=2, ensure_ascii=False), encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ordered(frame):
    # The persisted row order resolves same-month ties, including medication state.
    return frame.sort_values(["PATNO", "visit_date", "row_id"], kind="stable").reset_index(drop=True)


def reconstruct_targets(frame):
    """Reproduce original date-mean targets and retain their source date."""
    frame = frame.copy()
    date_means = frame.groupby(["PATNO", "visit_date"], sort=True).NP3TOT.mean()
    lookup = {}
    for pat, group in date_means.groupby(level=0):
        dates = group.index.get_level_values("visit_date").to_numpy()
        scores = group.to_numpy()
        for date in dates:
            days = (dates - date) / np.timedelta64(1, "D")
            values = {}
            for h, lo, hi, center in [(12, 250, 480, 365), (24, 550, 910, 730)]:
                valid = np.flatnonzero((days >= lo) & (days <= hi))
                j = valid[np.argmin(abs(days[valid] - center))] if len(valid) else None
                values[f"target_np3tot_{h}m"] = scores[j] if j is not None else np.nan
                values[f"target_date_{h}m"] = dates[j] if j is not None else pd.NaT
            lookup[pat, date] = values
    rebuilt = pd.DataFrame([lookup[p, np.datetime64(d, "ns")] for p, d in
                            zip(frame.PATNO, frame.visit_date)])
    for col in rebuilt:
        if col.startswith("target_np3tot"):
            if not np.allclose(frame[col], rebuilt[col], equal_nan=True):
                raise ValueError(f"Longitudinal target reconstruction differs: {col}")
        frame[col] = rebuilt[col].to_numpy()
    for h in [12, 24]:
        delta = frame[f"target_np3tot_{h}m"] - frame.NP3TOT
        labels = np.where(delta.notna(), (delta >= 3.5).astype(float), np.nan)
        if not np.allclose(frame[f"prog_mcid_{h}m"], labels, equal_nan=True):
            raise ValueError("Progression label disagrees with raw motor delta")
        frame[f"prog_mcid_{h}m"] = labels
    return frame


def latest_rows(frame, horizon):
    # Keep the historical training eligibility (a 12m label), then require the
    # horizon label. tail(1) keeps a complete row, including its missing values.
    valid = frame.dropna(subset=["target_np3tot_12m", f"target_np3tot_{horizon}m"])
    return ordered(valid).groupby("PATNO", sort=False).tail(1).reset_index(drop=True)


def evaluation_rows(frame):
    a, b = latest_rows(frame, 12), latest_rows(frame, 24)
    union = ordered(pd.concat([a, b]).drop_duplicates("row_id"))
    union["eval12"] = union.row_id.isin(a.row_id)
    union["eval24"] = union.row_id.isin(b.row_id)
    return union


def prepare(output):
    dest = Path(output) / "data"
    dest.mkdir(parents=True, exist_ok=False)
    patients, audits = {}, []
    source_hashes = {}
    canonical = pd.read_csv(DATA / "patient_split.csv")
    if canonical.PATNO.duplicated().any():
        raise ValueError("Duplicate patients in canonical split")
    expected_split = canonical.set_index("PATNO").SPLIT.str.lower()
    for split in ["train", "val", "test"]:
        path = DATA / f"{split}_longitudinal.parquet"
        frame = pd.read_parquet(path)
        frame["row_id"] = np.arange(len(frame))
        frame = reconstruct_targets(ordered(frame))
        if not frame.PATNO.map(expected_split).eq(split).all():
            raise ValueError("Canonical patient assignment mismatch")
        if not frame.split.eq(split).all():
            raise ValueError("Longitudinal split label mismatch")
        patients[split] = set(frame.PATNO)
        source_hashes[str(path)] = sha(path)
        oldpath = DATA / f"{split}_tabular_latest.parquet"
        old = pd.read_parquet(oldpath).sort_values("PATNO").reset_index(drop=True)
        source_hashes[str(oldpath)] = sha(oldpath)
        correct = latest_rows(frame, 12).sort_values("PATNO").reset_index(drop=True)
        if old.PATNO.tolist() != correct.PATNO.tolist() or not old.visit_date.equals(correct.visit_date):
            raise ValueError("Historical and corrected 12m index patients/dates differ")
        record = dict(split=split, patients=len(patients[split]), visits=len(frame),
                      samples12=len(correct), samples24=len(latest_rows(frame, 24)),
                      labelled_train_prefixes=int(frame[TARGETS[0]].notna().sum()),
                      prevalence12=float(correct.prog_mcid_12m.mean()))
        for h in [12, 24]:
            col = f"target_np3tot_{h}m"
            record[f"historical_n{h}"] = int(old[col].notna().sum())
            record[f"same_index_valid{h}"] = int(correct[col].notna().sum())
            record[f"stale_or_wrong{h}"] = int((~np.isclose(old[col], correct[col], equal_nan=True)).sum())
        record["mixed_feature_cells"] = int((~np.isclose(old[STATIC_FEATURES], correct[STATIC_FEATURES], equal_nan=True)).sum())
        audits.append(record)
        frame.to_parquet(dest / f"{split}_longitudinal.parquet", index=False)
        evaluation_rows(frame).to_parquet(dest / f"{split}_evaluation.parquet", index=False)
        for h in [12, 24]:
            latest_rows(frame, h).to_parquet(dest / f"{split}_tabular_{h}m.parquet", index=False)
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        if patients[a] & patients[b]:
            raise ValueError(f"Patient leakage {a}/{b}")
    pd.DataFrame(audits).to_csv(Path(output) / "target_audit.csv", index=False)
    write_json(Path(output) / "data_audit.json", {
        "splits_disjoint": True, "longitudinal_targets_reconstructed": True,
        "labels_equal_raw_delta_ge_3_5": True, "raw_data_available": Path("data/raw").exists(),
        "raw_provenance_certified": False,
        "limitation": "Event joins omit acquisition dates; latest demographics/education are not time-stamped. Raw files required for prospective leakage certification. Same-month order unresolved.",
        "protocol": "latest complete eligible row per horizon; 24m requires both labels, as in original training; unchanged PATNO partitions",
        "target_aggregation": "mean of all motor states on the future date; original windows and MCID retained",
        "source_sha256": source_hashes, "splits": audits})
    write_json(Path(output) / "prepared_sha256.json", {str(p.relative_to(output)): sha(p) for p in dest.glob("*.parquet")})
    return audits


class Preprocessor:
    def fit(self, frame):
        frame = ordered(frame)
        self.medians = frame[FEATURES].median().fillna(0.0)
        values = frame.groupby("PATNO")[FEATURES].ffill().fillna(self.medians)
        self.scaler = StandardScaler().fit(values)
        return self

    def transform(self, frame):
        frame = ordered(frame)
        values = frame.groupby("PATNO")[FEATURES].ffill().fillna(self.medians)
        scaled = self.scaler.transform(values).astype(np.float32)
        if not np.isfinite(scaled).all():
            raise ValueError("Nonfinite features")
        return frame, scaled

    def state(self):
        return dict(feature_cols=FEATURES, medians=self.medians.to_dict(),
                    mean=self.scaler.mean_.tolist(), scale=self.scaler.scale_.tolist())

    @classmethod
    def restore(cls, state):
        if state["feature_cols"] != FEATURES:
            raise ValueError("Checkpoint feature order mismatch")
        obj = cls()
        obj.medians = pd.Series(state["medians"])
        obj.scaler = StandardScaler()
        obj.scaler.mean_ = np.asarray(state["mean"])
        obj.scaler.scale_ = np.asarray(state["scale"])
        obj.scaler.var_ = obj.scaler.scale_ ** 2
        obj.scaler.n_features_in_ = len(FEATURES)
        obj.scaler.feature_names_in_ = np.asarray(FEATURES, dtype=object)
        return obj


def make_dataset(frame, prep, rows=None):
    frame, values = prep.transform(frame)
    if rows is None:
        rows = frame[frame[TARGETS[0]].notna()]
    positions = {rid: i for i, rid in enumerate(frame.row_id)}
    first = {p: g.index[0] for p, g in frame.groupby("PATNO", sort=False)}
    sequences, curs, t12, t24, cls, pats = [], [], [], [], [], []
    for row in rows.itertuples(index=False):
        end = positions[row.row_id]
        prefix = frame.iloc[first[row.PATNO]:end + 1]
        if prefix.PATNO.nunique() != 1 or prefix.visit_date.max() > row.visit_date:
            raise ValueError("Invalid future/patient prefix")
        if prefix.iloc[-1].NP3TOT != row.NP3TOT:
            raise ValueError("Wrong residual anchor")
        sequences.append(values[first[row.PATNO]:end + 1])
        curs.append(float(row.NP3TOT))  # NEVER take this from scaled values.
        t12.append(row.target_np3tot_12m)
        t24.append(row.target_np3tot_24m)
        cls.append(row.prog_mcid_12m)
        pats.append(row.PATNO)
    return historical.LongitudinalSequenceDataset(sequences, curs, t12, t24, cls, pats)
