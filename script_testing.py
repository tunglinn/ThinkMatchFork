import pandas as pd

matches = pd.read_csv('match_test.csv')
print(matches.columns)
match_dict = {}

for cat in matches.columns:
    match_dict[cat] = matches[cat].tolist()

print(match_dict)
