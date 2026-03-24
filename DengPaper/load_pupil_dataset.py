from scipy.io import loadmat
import numpy as np

def load_pupil_mat(path):
    data = loadmat(path, simplify_cells=True)

    if "Pupil_data_py" not in data:
        raise KeyError(f"Expected 'Pupil_data_py'. Found keys: {list(data.keys())}")

    sessions = data["Pupil_data_py"]

    if isinstance(sessions, dict):
        sessions = [sessions]
    elif isinstance(sessions, np.ndarray):
        sessions = sessions.tolist()

    return sessions