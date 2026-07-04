import numpy as np
import onnxruntime as ort
import torch
import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from src.datasets.base_dataset import BaseFaceMeshDataset


def collect_onnx_logits(onnx_path: str, hpo_model, val_dataset: BaseFaceMeshDataset):
    """Return {model: {'logit': np.ndarray, 'label': np.ndarray}} on ONNX outputs."""
    # Group images into a single batch of face meshes to reuse your preprocessing.
    data = [{'data': d['x'], 'label': d['y']} for d in val_dataset]
    input_tensor = torch.vstack([d['x'] for d in val_dataset])
    all_labels = torch.vstack([d['y'] for d in val_dataset]).numpy().astype(np.float32)
    all_age = torch.vstack([d['age'] for d in val_dataset]).numpy().astype(np.float32)
    all_gender = torch.vstack([d['gender'] for d in val_dataset]).numpy().astype(np.float32)
    all_ethnicity = torch.vstack([d['ethnicity'] for d in val_dataset]).numpy().astype(np.float32)
    n = len(data)

    if not hpo_model.is_trained():
        return None

    reduced = input_tensor.reshape(n, hpo_model.dimensions, -1).numpy().astype(np.float32)

    sess = ort.InferenceSession(onnx_path)
    logits = np.empty(n, dtype=np.float32)
    for i in tqdm.tqdm(range(n), desc="Calibrating ONNX model", total=len(all_labels)):
        out = sess.run(None, {
            "input": reduced[i:i + 1],
            "age": all_age[i],
            "gender": all_gender[i],
            "ethnicity": all_ethnicity[i],
        })[0]
        logits[i] = out.flatten()[0]

    labels = np.asarray([lbl.item() for lbl in all_labels], dtype=float)
    keep = ~np.isnan(labels)
    return {
        "logit": logits[keep],
        "label": labels[keep].astype(int),
    }


def fit_beta(logits: np.ndarray, y: np.ndarray):
    """Beta calibration: p_cal = sigmoid(a*log(p) - b*log(1-p) + c)."""
    p = 1.0 / (1.0 + np.exp(-logits))
    eps = 1e-6
    p = np.clip(p, eps, 1.0 - eps)
    X = np.column_stack([np.log(p), -np.log(1.0 - p)])
    clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(X, y)
    a, b = clf.coef_[0]
    c = float(clf.intercept_[0])
    p_cal = clf.predict_proba(X)[:, 1]
    return {"method": "beta", "a": float(a), "b": float(b), "c": float(c)}, p_cal


def fit_temperature(logits: np.ndarray, y: np.ndarray):
    """Temperature scaling via LBFGS on BCE-with-logits."""
    z = torch.tensor(logits, dtype=torch.float)
    yt = torch.tensor(y, dtype=torch.float)
    T = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=200)

    def loss_fn():
        opt.zero_grad()
        l = torch.nn.functional.binary_cross_entropy_with_logits(z / T, yt)
        l.backward()
        return l

    opt.step(loss_fn)
    T_val = float(T.detach().item())
    p_cal = 1.0 / (1.0 + np.exp(-logits / T_val))
    return {"method": "temperature", "T": T_val}, p_cal


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error using equal-width bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    err = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        err += (m.sum() / len(p)) * abs(p[m].mean() - y[m].mean())
    return float(err)


def calibrate_all(data: {str: np.ndarray, str: np.ndarray}, min_positives_for_beta: int = 30):
    logits, y = data["logit"], data["label"]
    p_raw = 1.0 / (1.0 + np.exp(-logits))

    n_pos = int(y.sum())
    n_neg = int((1 - y).sum())

    # Fall back to temperature scaling when there aren't enough positives
    # (or negatives) to reliably fit the 3-parameter beta model.
    can_beta = n_pos >= min_positives_for_beta and n_neg >= min_positives_for_beta
    candidates = []
    try:
        candidates.append(fit_temperature(logits, y))
    except Exception as e:
        print(f"temperature scaling failed: {e}")
    if can_beta:
        try:
            candidates.append(fit_beta(logits, y))
        except Exception as e:
            print(f"beta calibration failed: {e}")

    # Pick calibrator with the lowest Brier score, but only accept it if
    # it improves over the raw sigmoid — otherwise the val set is too small.
    brier_raw = brier_score_loss(y, p_raw)
    best = None
    best_brier = brier_raw
    for params, p_cal in candidates:
        b = brier_score_loss(y, p_cal)
        if b < best_brier:
            best_brier = b
            best = (params, p_cal)

    if best is None:
        p_final = p_raw
        method = "none"
    else:
        p_final = best[1]
        method = best[0]["method"]

    result = {
        "method": method,
        "n": len(y),
        "n_pos": n_pos,
        "brier_raw": brier_raw,
        "brier_cal": best_brier,
        "ece_raw": ece(p_raw, y),
        "ece_cal": ece(p_final, y),
    }
    if best:
        result.update(best[0])
    return result
