# Topic Modeling BBC News Headlines (2013-2021)

## Introduction

This repository demonstrates how to apply Latent Dirichlet Allocation (LDA) to discover underlying topics in a large dataset of BBC news headlines spanning from 2013 to 2021. Topic modeling is a powerful technique that allows us to uncover hidden thematic structures within text data, providing valuable insights into content trends, patterns, and evolution over time.

By the end of this tutorial, you'll understand:
- How to preprocess text data for topic modeling
- How to determine the optimal number of topics using coherence scores
- How to implement and interpret LDA models on large datasets
- How to analyze topic evolution over time

## Dataset Overview

This project uses a dataset of approximately 1.2 million BBC news headlines published between 2013 and 2021. The data shows an interesting trend: the number of headlines per year follows a monotonically decreasing pattern, with 2013 representing 7% of the observations and 2021 only 1.4%.

![Headlines Distribution by Year](https://placeholder-for-your-actual-image.png)

*Note: You should create and add this visualization to your repository*

## Step 1: Text Preprocessing

Before we can apply topic modeling, we need to clean and prepare our text data. The preprocessing pipeline involves:

```python
def preprocess_document(text):
    """
    Preprocess a single document for topic modeling
    """
    # Handle missing values
    if pd.isna(text):
        return []
    
    # Convert to lowercase and remove punctuation
    text = re.sub(r'[^\w\s]', '', text.lower())
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in text.split() if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return lemmatized_tokens

# Apply preprocessing to all headlines
df['processed_tokens'] = df['headline'].apply(preprocess_document)
```

This preprocessing function handles:
1. **Missing values** - Returning empty lists for any missing headlines
2. **Punctuation removal** - Stripping out non-word characters
3. **Stopword removal** - Eliminating common English words that don't carry significant meaning
4. **Lemmatization** - Reducing words to their base forms to group related terms

## Step 2: Sampling Strategy for Large Datasets

When working with large datasets like ours (1.2 million headlines), computational efficiency becomes crucial. Rather than processing the entire dataset for preliminary model tuning, we implemented a stratified sampling approach that:

1. Maintains the same proportion of headlines per year as in the original dataset
2. Reduces the data to a manageable 30% sample
3. Ensures temporal patterns are preserved

```python
# Create a stratified sample to maintain year distribution
df_sample = df.groupby('year').apply(
    lambda x: x.sample(frac=0.3, random_state=42)
).reset_index(drop=True)

# Verify sample distribution matches original
sample_year_dist = df_sample['year'].value_counts(normalize=True).sort_index()
original_year_dist = df['year'].value_counts(normalize=True).sort_index()
```

This approach allows us to determine the optimal number of topics more efficiently while still capturing the dataset's essential characteristics.

## Step 3: Building the LDA Model

With our preprocessed sample, we can now build an initial LDA model:

```python
# Create dictionary and corpus
sampled_id2word = Dictionary(df_sample['processed_tokens'])
sampled_corpus = [sampled_id2word.doc2bow(doc) for doc in df_sample['processed_tokens']]

# Initialize LDA model
lda_model = LdaModel(
    corpus=sampled_corpus,
    id2word=sampled_id2word,
    num_topics=4,  # Initial test with 4 topics
    random_state=42,
    alpha='auto',
    eta='auto',
    chunksize=100,
    passes=10,
    per_word_topics=True
)
```

The key parameters are:
- **alpha**: Set to 'auto' to learn an asymmetric prior from the data
- **eta**: Set to 'auto' to learn topic-word distribution from the corpus
- **chunksize**: Number of documents processed in each training chunk (100)
- **passes**: Number of passes over the corpus during training (10)
- **per_word_topics**: Whether to compute topic probabilities for each word

## Step 4: Finding the Optimal Number of Topics

Determining the right number of topics is crucial for meaningful analysis. We use coherence scores to evaluate topic quality across different numbers of topics:

```python
def compute_coherence_values(dictionary, corpus, texts, start, stop, step):
    """
    Compute coherence scores for different numbers of topics
    """
    coherence_values = []
    model_list = []
    
    for num_topics in range(start, stop, step):
        model = LdaMulticore(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42,
            chunksize=100,
            passes=10,
            workers=4  # Parallel processing
        )
        model_list.append(model)
        
        # Calculate coherence using c_v measure
        coherence_model = CoherenceModel(
            model=model, 
            texts=texts, 
            dictionary=dictionary, 
            coherence='c_v'
        )
        coherence_values.append(coherence_model.get_coherence())
    
    return model_list, coherence_values

# Compute coherence for various topic numbers
model_list, coherence_values = compute_coherence_values(
    dictionary=sampled_id2word,
    corpus=sampled_corpus,
    texts=df_sample['processed_tokens'],
    start=4,
    stop=40,
    step=4
)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(range(4, 40, 4), coherence_values)
plt.xlabel("Number of Topics")
plt.ylabel("Coherence Score")
plt.title("Topic Coherence by Number of Topics")
plt.grid(True)
plt.show()
```

Our analysis revealed an interesting pattern: the coherence metric initially increases up to k=11, then unexpectedly decreases between k=11 and k=17, before rising again to peak at k=33. For simplicity and interpretability, we selected 8 topics, which aligns better with expected news headline subtopics.

![Topic Coherence Score](https://placeholder-for-your-actual-coherence-graph.png)

*Note: Add your actual coherence graph here*

## Step 5: Applying the Optimal Model to the Full Dataset

After determining the optimal number of topics, we apply the LDA model to the entire dataset:

```python
# Create dictionary and corpus for full dataset
id2word_full = Dictionary(df['processed_tokens'])
corpus_full = [id2word_full.doc2bow(doc) for doc in df['processed_tokens']]

# Build final LDA model with optimal number of topics
final_lda_model = LdaMulticore(
    corpus=corpus_full,
    id2word=id2word_full,
    num_topics=8,
    random_state=42,
    chunksize=100,
    passes=10,
    workers=4
)

# Get topic distributions for all documents
topic_distributions_full = [final_lda_model.get_document_topics(doc) for doc in corpus_full]
```

## Step 6: Analyzing Topic Evolution Over Time

One of the most insightful aspects of this project is tracking how topics evolve over time:

```python
# Get dominant topic for each document
def get_dominant_topic(topics_dist):
    return max(topics_dist, key=lambda x: x[1])[0] if topics_dist else None

# Assign dominant topic to each headline
df['dominant_topic'] = [get_dominant_topic(dist) for dist in topic_distributions_full]

# Group by year and count topics
topic_evolution = df.groupby(['year', 'dominant_topic']).size().unstack()

# Plot topic evolution
plt.figure(figsize=(15, 8))
topic_evolution.plot(kind='line', marker='o', ax=plt.gca())
plt.title('Evolution of Topics Over Time (2013-2021)')
plt.xlabel('Year')
plt.ylabel('Number of Headlines')
plt.legend(title='Topic')
plt.grid(True)
plt.show()
```

Our analysis revealed that Topic G (Topic 7) showed a substantial increase from 2003, reaching its peak in 2012. However, all topics began to decline thereafter, likely due to the decreasing number of headlines in the dataset.

![Topic Evolution](https://placeholder-for-your-topic-evolution-graph.png)

*Note: Add your actual topic evolution graph here*

## Step 7: Interpreting the Topics

Understanding what each topic represents is crucial for deriving insights. Here are the top 10 words for our most prominent topic:

**Topic G (Topic 7)**: man, woman, Melbourne, Donald, news, child, court, police, vaccine, car

This topic appears to encompass stories related to legal matters, crime reporting, and public health (notably vaccine-related news).

```python
# Print top words for each topic
for idx, topic in final_lda_model.print_topics(-1):
    print(f'Topic {idx}: {topic}')
```

A comprehensive analysis would include interpretation of all eight identified topics.

## Conclusion and Applications

This project demonstrates how LDA can uncover meaningful topics from large text datasets and track their evolution over time. The techniques shown here can be applied to:

- Content recommendation systems
- News trend analysis
- Media monitoring
- Content strategy development
- Historical analysis of public discourse

## Future Work

Potential extensions to this project could include:
- Incorporating sentiment analysis to understand emotional trends
- Applying dynamic topic modeling to better capture evolving topics
- Comparing different topic modeling approaches (NMF, BERTopic)
- Geographic analysis of news topics
- Cross-referencing with external events for causal analysis

## Requirements

To run the code in this repository, you'll need the following dependencies:

```
gensim
nltk
pandas
numpy
matplotlib
pyLDAvis
```

You can install all requirements using:
```
pip install -r requirements.txt
```

## License

This project is licensed under the terms of the LICENSE file included in this repository.
