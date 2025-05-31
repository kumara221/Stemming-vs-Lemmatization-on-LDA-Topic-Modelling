# Stemming-vs-Lemmatizati-on-LDA-Topic-Modelling

📌 Project Description

This project explores the impact of preprocessing techniques—specifically stemming and lemmatization—on the performance of Latent Dirichlet Allocation (LDA) topic modeling using Gibbs Sampling. Using a dataset of scientific publication abstracts from 2013 to 2025, we aim to uncover underlying research trends over the years and evaluate how linguistic preprocessing influences the coherence and efficiency of topic modeling.

The LDA implementation is done manually using NumPy for full algorithmic control and performance transparency. The model was applied year-by-year to JSON-structured data, with coherence scores (UMass-like) used to assess the quality of topics generated under each preprocessing approach. The project emphasizes not only the outcome of topic discovery but also the computational efficiency and interpretability resulting from preprocessing choices.

🎯 Project Objectives

* Compare stemming and lemmatization in LDA topic modeling.
* Evaluate topic coherence and processing time for each method.
* Analyze research trends in abstracts over a 13-year period.
* Provide insight into the trade-off between linguistic accuracy and computational cost.

📊 Key Findings

* Stemming outperformed lemmatization in both coherence scores and processing time.
* Lemmatization, though linguistically richer, often led to less coherent topics and longer runtimes.
* Topic coherence is strongly influenced by the volume of data per year.
* Gibbs Sampling requires well-preprocessed input data to yield meaningful topic distributions.

✅ Project Conclusion

The results clearly show that stemming is more efficient and effective for LDA topic modeling in scenarios involving large-scale temporal data. While lemmatization has linguistic advantages, its practical benefit in unsupervised topic modeling is limited when weighed against coherence and runtime.

This study reinforces the importance of choosing appropriate preprocessing methods tailored to the task objective—especially in large-scale text analysis where balance between precision and performance is critical.

🔍 Future Work

* Experiment with advanced topic models like BERTopic or NMF with contextual embeddings.
* Introduce dynamic topic modeling to track topic evolution over time.
* Build interactive visualizations to explore annual topic shifts.
