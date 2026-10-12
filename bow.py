from split_utils import SEED, split_nyt, tokenize_nltk, doc_avg_embedding, load_glove
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

def tokenize_nltk(text):
    return word_tokenize(text.lower())

# -------- Task1‑1 Binary Bag of Words --------
# binary=True：1出现/0不出现
vec_binary = CountVectorizer(tokenizer=tokenize_nltk, binary=True, token_pattern=None)
X_train_bin = vec_binary.fit_transform(df_train["text"])
X_test_bin = vec_binary.transform(df_test["text"])

clf_bin = LogisticRegression(max_iter=2000)
clf_bin.fit(X_train_bin, df_train["label"])
y_pred_bin = clf_bin.predict(X_test_bin)

acc_bin = accuracy_score(df_test["label"], y_pred_bin)
f1_macro_bin = f1_score(df_test["label"], y_pred_bin, average="macro")
print("Binary BoW Acc:", acc_bin)
print("Binary BoW Macro‑F1:", f1_macro_bin)

# -------- Task1‑2 Word Frequency 词频BoW --------
vec_freq = CountVectorizer(tokenizer=tokenize_nltk, binary=False, token_pattern=None)
X_train_freq = vec_freq.fit_transform(df_train["text"])
X_test_freq = vec_freq.transform(df_test["text"])

clf_freq = LogisticRegression(max_iter=2000)
clf_freq.fit(X_train_freq, df_train["label"])
y_pred_freq = clf_freq.predict(X_test_freq)

acc_freq = accuracy_score(df_test["label"], y_pred_freq)
f1_macro_freq = f1_score(df_test["label"], y_pred_freq, average="macro")
print("Word Freq BoW Acc:", acc_freq)
print("Word Freq BoW Macro‑F1:", f1_macro_freq)