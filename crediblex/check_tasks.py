import os, pandas as pd

csv_path = "training_data.csv"
df = pd.read_csv(csv_path)
rows = len(df)
t1 = "PASS" if rows >= 10000 else "FAIL"
print("TASK1:" + str(rows) + ":" + t1)

var = df["fact_score"].var() if "fact_score" in df.columns else -1
t2a = "PASS" if var > 0 else "FAIL"
print("TASK2a_var:" + str(round(var,4)) + ":" + t2a)

if "emotion_label" in df.columns:
    mn = int(df["emotion_label"].min())
    mx = int(df["emotion_label"].max())
    t2b = "PASS" if mn >= 0 and mx <= 27 else "FAIL"
    print("TASK2b_range:" + str(mn) + "-" + str(mx) + ":" + t2b)

print("TASK3_smoke_test:" + str(os.path.exists("smoke_test.py")))
print("TASK4_train:" + str(os.path.exists("train.py")))

pth_exists = os.path.exists("model_v1.pth")
pth_size = round(os.path.getsize("model_v1.pth") / 1024 / 1024, 1) if pth_exists else 0
print("TASK5_model:" + str(pth_exists) + ":" + str(pth_size) + "MB")
