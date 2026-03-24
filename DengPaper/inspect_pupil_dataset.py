from pprint import pprint
from load_pupil_dataset import load_pupil_mat

sessions = load_pupil_mat("Pupil_dataset_py.mat")

print("Number of sessions:", len(sessions))

s0 = sessions[0]
print("\nSession keys:")
pprint(list(s0.keys()))

print("\nSubject:", s0["Subject"])
print("Age:", s0["Age"])
print("Group:", s0["Group"])

te = s0["Task_epocs"]
print("\nTask_epocs type:", type(te))

if isinstance(te, dict):
    print("Task_epocs keys:")
    pprint(list(te.keys()))
    for k, v in te.items():
        print(f"{k}: {type(v)}")
        try:
            print(f"  len={len(v)}")
        except:
            pass
elif isinstance(te, list):
    print("Task_epocs length:", len(te))
    if len(te) > 0:
        print("First row type:", type(te[0]))
        print("First row:", te[0])

print("\nTask_data:", s0["Task_data"])
print("Wisc type:", type(s0["Wisc"]))