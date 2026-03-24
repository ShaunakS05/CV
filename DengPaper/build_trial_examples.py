import numpy as np
import pandas as pd
from load_pupil_dataset import load_pupil_mat


def group_to_label(group_name: str) -> int:
    """
    Binary task:
      Ctrl -> 0
      on-ADHD / off-ADHD -> 1
    """
    return 0 if str(group_name).strip() == "Ctrl" else 1


def normalize_trial_pupil(pupil_vec):
    """
    Convert a numeric sequence to a clean 1D float32 array.
    Returns None if nothing usable is found.
    """
    try:
        x = np.array(pupil_vec).reshape(-1)
    except Exception:
        return None

    clean = []
    for v in x:
        try:
            fv = float(v)
            if np.isfinite(fv):
                clean.append(fv)
        except Exception:
            pass

    if len(clean) == 0:
        return None

    return np.asarray(clean, dtype=np.float32)


def _collect_numeric_arrays(obj):
    """
    Recursively collect numeric arrays from nested MATLAB/Python containers.
    """
    out = []

    if obj is None:
        return out

    if isinstance(obj, dict):
        for v in obj.values():
            out.extend(_collect_numeric_arrays(v))
        return out

    if isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(_collect_numeric_arrays(v))
        return out

    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            for v in obj.flat:
                out.extend(_collect_numeric_arrays(v))
            return out
        arr = normalize_trial_pupil(obj)
        if arr is not None and len(arr) > 0:
            out.append(arr)
        return out

    arr = normalize_trial_pupil(obj)
    if arr is not None and len(arr) > 0:
        out.append(arr)

    return out


def extract_pupil_signal(pupil_entry):
    """
    Recover the actual pupil signal from nested structures.

    In this dataset, Pupil is often [time_vector, pupil_vector].
    We explicitly prefer the non-monotonic vector.
    """
    if pupil_entry is None:
        return None

    def is_strictly_increasing_by_one(arr):
        if arr is None or len(arr) < 5:
            return False
        diffs = np.diff(arr[: min(len(arr), 200)])
        return np.allclose(diffs, 1.0)

    # Case 1: list/tuple with two entries
    if isinstance(pupil_entry, (list, tuple)) and len(pupil_entry) == 2:
        a = normalize_trial_pupil(pupil_entry[0])
        b = normalize_trial_pupil(pupil_entry[1])

        # If one looks like a time vector, use the other
        if a is not None and is_strictly_increasing_by_one(a):
            return b
        if b is not None and is_strictly_increasing_by_one(b):
            return a

        # fallback: use second
        if b is not None and len(b) > 0:
            return b
        return a

    # Case 2: object ndarray with two entries
    if isinstance(pupil_entry, np.ndarray) and pupil_entry.dtype == object:
        flat = list(pupil_entry.flat)
        if len(flat) == 2:
            a = normalize_trial_pupil(flat[0])
            b = normalize_trial_pupil(flat[1])

            if a is not None and is_strictly_increasing_by_one(a):
                return b
            if b is not None and is_strictly_increasing_by_one(b):
                return a

            if b is not None and len(b) > 0:
                return b
            return a

    # General recursive fallback
    candidates = _collect_numeric_arrays(pupil_entry)
    candidates = [c for c in candidates if c is not None and len(c) >= 20]

    if not candidates:
        return None

    # Remove obvious time vectors
    non_time = [c for c in candidates if not is_strictly_increasing_by_one(c)]
    if non_time:
        return max(non_time, key=len).astype(np.float32)

    # fallback
    return max(candidates, key=len).astype(np.float32)


def rows_from_task_epocs(task_epocs):
    """
    Convert dict-of-columns into list-of-rows.
    """
    if not isinstance(task_epocs, dict):
        raise ValueError(f"Unsupported Task_epocs type: {type(task_epocs)}")

    keys = list(task_epocs.keys())
    n = None

    for k in keys:
        v = task_epocs[k]
        if isinstance(v, (list, tuple, np.ndarray)):
            n = len(v)
            break

    if n is None:
        raise ValueError("Could not infer trial count from Task_epocs")

    rows = []
    for i in range(n):
        row = {}
        for k in keys:
            v = task_epocs[k]
            if isinstance(v, (list, tuple, np.ndarray)):
                row[k] = v[i]
            else:
                row[k] = v
        rows.append(row)

    return rows


def to_int_safe(x, default=-1):
    try:
        return int(float(x))
    except Exception:
        return default


def to_float_safe(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def build_examples(mat_path):
    """
    Build a trial-level DataFrame from Pupil_dataset_py.mat.

    Output columns:
      session_idx
      subject_id
      age
      group_name
      label
      trial
      load
      distractor
      perform
      rtime
      pupil
    """
    sessions = load_pupil_mat(mat_path)
    rows = []

    for sess_idx, sess in enumerate(sessions):
        subject = to_int_safe(sess.get("Subject"))
        age = to_float_safe(sess.get("Age"))
        group = str(sess.get("Group"))
        label = group_to_label(group)

        task_epocs = sess.get("Task_epocs")
        if task_epocs is None:
            continue

        trial_rows = rows_from_task_epocs(task_epocs)

        for row in trial_rows:
            pupil = extract_pupil_signal(row.get("Pupil"))
            if pupil is None or len(pupil) < 20:
                continue

            rows.append({
                "session_idx": sess_idx,
                "subject_id": subject,
                "age": age,
                "group_name": group,
                "label": label,
                "trial": to_int_safe(row.get("Trial")),
                "load": to_int_safe(row.get("Load")),
                "distractor": row.get("Distractor", ""),
                "perform": to_int_safe(row.get("Perform")),
                "rtime": to_float_safe(row.get("Rtime")),
                "pupil": pupil,
            })

    return pd.DataFrame(rows)