import os
import numpy as np
import pandas as pd

RESULTS_DIR = "results/metrics"
UQ_FILE = os.path.join(RESULTS_DIR, "uq_data.npy")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "uq_results.csv")

# Load UQ data
uq_values = np.load(UQ_FILE, allow_pickle=True).item()

# Prepare DataFrame
uq_summary = []
for env_name, values in uq_values.items():
    uq_summary.append([
        env_name,
        np.mean(values),
        np.max(values),
        np.median(values)
    ])

df_uq = pd.DataFrame(
    uq_summary,
    columns=["Environment", "Mean Policy Variance", "Max Policy Variance", "Median Policy Variance"]
)

# Save results
df_uq.to_csv(OUTPUT_FILE, index=False)
print(f"UQ results saved to {OUTPUT_FILE}")
