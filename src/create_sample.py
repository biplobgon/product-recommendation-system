import pandas as pd
import os

# Ensure folder exists
os.makedirs("data/sample", exist_ok=True)

# Load full dataset
events = pd.read_csv("data/raw/events.csv")

# Take sample (10K rows)
sample = events.sample(10000, random_state=42)

# Save sample
sample.to_csv("data/sample/sample_events.csv", index=False)

print("Sample dataset created successfully!")
