import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


import pandas as pd
from sklearn.model_selection import train_test_split

def split_nyt(df):
    # 打乱
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    train, temp = train_test_split(df, test_size=0.2, random_state=SEED)
    val, test = train_test_split(temp, test_size=0.5, random_state=SEED)
    print(f"train:{len(train)}, val:{len(val)}, test:{len(test)}")
    return train, val, test

import pandas as pd
SEED = 42


df_nyt = pd.read_csv("nyt.csv")

print(df_nyt[["text","label"]].head())

df_train, df_val, df_test = split_nyt(df_nyt)

