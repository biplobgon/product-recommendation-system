import pandas as pd

events = pd.read_csv("data/raw/events.csv")

print(events.head())
print(events.info())
print(events['event'].value_counts())
