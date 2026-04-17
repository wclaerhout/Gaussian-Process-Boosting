# from more_itertools import sample
import pandas as pd
import numpy as np
import random

df = pd.read_excel('data/my_data.xlsx')
print(df.head())
print(df.columns)

cortisol_values = df["cortisol"]
features = df[["hours_since_waking", "time_of_day", "sleep_duration", 
               "physical_activity", "food_intake", "caffeine_intake"]]
features["caffeine_intake"] = [1 if x == True else 0 for x in features["caffeine_intake"]]

train_idx = random.sample(range(len(df)), int(0.8 * len(df)))
X_train = features.iloc[train_idx]
y_train = cortisol_values.iloc[train_idx]
X_test = features.drop(train_idx)
y_test = cortisol_values.drop(train_idx)
print(cortisol_values.head())
print(features.head())
print(len(X_train), len(y_train), len(X_test), len(y_test))