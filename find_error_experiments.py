import pandas as pd

df = pd.read_csv("experiment_results_smoke_1/run_status.csv")

failed = df[df["Status"] != "OK"]

print(failed[
    [
        "Perfil",
        "L",
        "Run",
        "Seed",
        "Status",
        "Stage",
        "Wall time (s)",
        "Error type",
        "Error message",
    ]
].to_string(index=False))