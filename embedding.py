from split_utils import SEED, split_nyt, tokenize_nltk, doc_avg_embedding, load_glove

def load_glove(path="glove.6B.100d.txt"):
    glove_dict = {}
    with open(path, encoding="utf‑8") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vec = np.array([float(x) for x in parts[1:]])
            glove_dict[word] = vec
    return glove_dict

glove_wv = load_glove()

X_train_glove = build_doc_emb_matrix(df_train, glove_wv)
X_test_glove = build_doc_emb_matrix(df_test, glove_wv)

clf_glove = LogisticRegression(max_iter=3000)
clf_glove.fit(X_train_glove, df_train["label"])
y_pred_glove = clf_glove.predict(X_test_glove)

acc_glove = accuracy_score(df_test["label"], y_pred_glove)
f1_glove = f1_score(df_test["label"], y_pred_glove, average="macro")
print("GloVe Acc", acc_glove)
print("GloVe Macro‑F1", f1_glove)

from datasets import load_dataset
from gensim.models import Word2Vec


df_ag = pd.read_csv("ag.csv")
ag_sentences = []
for txt in df_ag["text"]:
    tokens = tokenize_nltk(txt)
    ag_sentences.append(tokens)

# 训练Word2Vec，100维
w2v_ag = Word2Vec(sentences=ag_sentences, vector_size=100, window=5, min_count=2, workers=4, seed=SEED)
wv_ag = w2v_ag.wv

X_train_w2vag = build_doc_emb_matrix(df_train, wv_ag)
X_test_w2vag = build_doc_emb_matrix(df_test, wv_ag)

clf_w2vag = LogisticRegression(max_iter=3000)
clf_w2vag.fit(X_train_w2vag, df_train["label"])
y_pred_w2vag = clf_w2vag.predict(X_test_w2vag)

acc_w2vag = accuracy_score(df_test["label"], y_pred_w2vag)
f1_w2vag = f1_score(df_test["label"], y_pred_w2vag, average="macro")
print("Word2Vec‑AGNews Acc", acc_w2vag)
print("Word2Vec‑AGNews Macro‑F1", f1_w2vag)

nyt_sentences = []
for txt in df_train["text"]:
    tokens = word_tokenize(txt.lower())
    nyt_sentences.append(tokens)

w2v_nyt = Word2Vec(sentences=nyt_sentences, vector_size=100, window=5, min_count=2, workers=4, seed=SEED)
wv_nyt = w2v_nyt.wv

X_train_w2vnyt = build_doc_emb_matrix(df_train, wv_nyt)
X_test_w2vnyt = build_doc_emb_matrix(df_test, wv_nyt)

clf_w2vnyt = LogisticRegression(max_iter=3000)
clf_w2vnyt.fit(X_train_w2vnyt, df_train["label"])
y_pred_w2vnyt = clf_w2vnyt.predict(X_test_w2vnyt)

acc_w2vnyt = accuracy_score(df_test["label"], y_pred_w2vnyt)
f1_w2vnyt = f1_score(df_test["label"], y_pred_w2vnyt, average="macro")
print("Word2Vec‑NYT Acc", acc_w2vnyt)
print("Word2Vec‑NYT Macro‑F1", f1_w2vnyt)