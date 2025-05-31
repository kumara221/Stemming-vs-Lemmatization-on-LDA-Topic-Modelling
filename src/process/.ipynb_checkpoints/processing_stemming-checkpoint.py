import os
import json
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from pathlib import Path
import random
from concurrent.futures import ProcessPoolExecutor
import time
from collections import defaultdict
import math

# Inisialisasi path untuk stemming
INPUT_DIR = Path('./../../data/preprocessing/stemming')
OUTPUT_DIR = Path('./../../output/lda_stemming')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Fungsi LDA Gibbs Sampling (sama)
def lda_gibbs(docs, K=3, alpha=0.1, beta=0.01, iterations=100):
    vocab = sorted({w for doc in docs for w in doc})
    W = len(vocab)
    word2id = {w: i for i, w in enumerate(vocab)}
    D = len(docs)

    doc_topic = np.zeros((D, K), dtype=int)
    topic_word = np.zeros((K, W), dtype=int)
    topic_sum = np.zeros(K, dtype=int)
    Z = []

    for d, doc in enumerate(docs):
        z_d = []
        for w in doc:
            t = random.randrange(K)
            z_d.append(t)
            doc_topic[d, t] += 1
            topic_word[t, word2id[w]] += 1
            topic_sum[t] += 1
        Z.append(z_d)

    for _ in range(iterations):
        for d, doc in enumerate(docs):
            for i, w in enumerate(doc):
                t_old = Z[d][i]
                wid = word2id[w]
                doc_topic[d, t_old] -= 1
                topic_word[t_old, wid] -= 1
                topic_sum[t_old] -= 1

                p_z = (doc_topic[d] + alpha) * (topic_word[:, wid] + beta) / (topic_sum + beta * W)
                p_z = p_z / p_z.sum()
                t_new = np.random.choice(K, p=p_z)

                Z[d][i] = t_new
                doc_topic[d, t_new] += 1
                topic_word[t_new, wid] += 1
                topic_sum[t_new] += 1

    top_words = {
        k: [vocab[i] for i in topic_word[k].argsort()[-10:][::-1]]
        for k in range(K)
    }
    return top_words, vocab

# Fungsi menghitung coherence (sama)
def compute_coherence(top_words, docs, epsilon=1e-12):
    doc_freq = defaultdict(int)
    for doc in docs:
        unique_words = set(doc)
        for word in unique_words:
            doc_freq[word] += 1

    coherence_scores = []
    for topic, words in top_words.items():
        score = 0.0
        for i in range(1, len(words)):
            for j in range(i):
                w_i = words[i]
                w_j = words[j]
                D_wi_wj = sum(1 for doc in docs if w_i in doc and w_j in doc)
                score += math.log((D_wi_wj + epsilon) / doc_freq.get(w_j, 1))
        coherence_scores.append(score)

    return sum(coherence_scores) / len(coherence_scores)

# Fungsi utama per tahun
def process_year(path_str):
    import matplotlib
    matplotlib.use('Agg')

    path = Path(path_str)
    year = path.stem.split("_")[-1]
    out_year_dir = OUTPUT_DIR / year
    out_year_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n→ Memproses tahun {year}")

    with open(path, 'r', encoding='utf-8') as f:
        records = json.load(f)

    docs = [rec["abstract"].split() for rec in records if rec.get("abstract", "").strip()]
    if not docs:
        print(f"⚠️ Tidak ada abstrak valid untuk {year}, dilewati.")
        return

    start_time = time.time()

    top_words, vocab = lda_gibbs(docs, K=3, iterations=100)
    coherence = compute_coherence(top_words, docs)
    print(f"✔️ Coherence Score (UMass-like) untuk {year}: {coherence:.4f}")

    for k, words in top_words.items():
        freqs = [sum(doc.count(w) for doc in docs) for w in words]

        # Bar chart
        plt.figure(figsize=(6, 4))
        plt.barh(words[::-1], freqs[::-1])
        plt.title(f"[{year}] Topik #{k+1}")
        plt.xlabel("Frekuensi")
        plt.tight_layout()
        plt.savefig(out_year_dir / f"bar_topic{(k+1):02d}_{year}.png")
        plt.close()

        # Word cloud
        wc = WordCloud(width=800, height=400, background_color='white')
        wc.generate_from_frequencies(dict(zip(words, freqs)))
        wc.to_file(out_year_dir / f"wordcloud_topic{(k+1):02d}_{year}.png")

    end_time = time.time()
    print(f"⏱️ Waktu proses {year}: {end_time - start_time:.2f} detik")

# Main runner paralel
if __name__ == "__main__":
    json_files = sorted(INPUT_DIR.glob("preprocessed_abstracts_stemming_*.json"))
    paths = [str(p) for p in json_files]

    with ProcessPoolExecutor(max_workers=16) as executor:
        executor.map(process_year, paths)
