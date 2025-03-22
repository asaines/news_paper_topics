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

## Step 1: Sampling Strategy for Large Datasets

When working with large datasets like ours (1.2 million headlines), computational efficiency becomes crucial. Rather than processing the entire dataset for preliminary model tuning, we implemented a stratified sampling approach that:

1. Maintains the same proportion of headlines per year as in the original dataset
2. Reduces the data to a manageable 30% sample
3. Ensures temporal patterns are preserved

```python
# Create a stratified sample to maintain year distribution
df_remaining, df_sample = train_test_split(df, test_size=0.3, random_state=42, stratify=df['year'])

```

This approach allows us to determine the optimal number of topics more efficiently while still capturing the dataset's essential characteristics.

## Step 2: Text Preprocessing

Before we can apply topic modeling, we need to clean and prepare our text data. The preprocessing pipeline involves:

```python
# Function to preprocess and tokenize a document
def preprocess_document(doc):
    # Get the default punctuation characters from the string module
    punctuation_pattern = f"[{re.escape(string.punctuation)}]"

    # Handle NA values
    doc = '' if pd.isnull(doc) else doc

    # Remove punctuation using the default pattern
    doc = re.sub(punctuation_pattern, '', doc)

    # Word tokenization using the contractions library
    doc = contractions.fix(doc)
    tokens = doc.split()
    
    #stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.lower() not in stop_words]


    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens

# Apply preprocessing to all headlines
df_sample['processed_tokens'] = df_sample['headline_text'].apply(preprocess_document)
```

This preprocessing function handles:
1. **Missing values** - Returning empty lists for any missing headlines
2. **Punctuation removal** - Stripping out non-word characters
3. **Stopword removal** - Eliminating common English words that don't carry significant meaning
4. **Lemmatization** - Reducing words to their base forms to group related terms


## Step 3: Building the LDA Model

With our preprocessed sample, we can now build an initial LDA model:

```python
# Create dictionary and corpus
sampled_id2word = corpora.Dictionary(df_sample['processed_tokens'])
sampled_corpus = [sampled_id2word.doc2bow(text) for text in df_sample['processed_tokens']]


# Initialize LDA model
lda_model = LdaModel(
    corpus=sampled_corpus,
    id2word=sampled_id2word,
    num_topics=4,  # Initial test with 4 topics
    random_state=42,
    alpha='auto',
    eta='auto',
    chunksize=100,
    random_state=100,
    passes=10,
    update_every=1,
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
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various numbers of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective numbers of topics
    """
    coherence_values = []
    model_list = []
    
    for num_topics in range(start, limit, step):
        # Build LDA model
        model = gensim.models.ldamulticore.LdaMulticore(workers=6, 
                                                        corpus=corpus, 
                                                        id2word=dictionary, 
                                                        num_topics=num_topics, 
                                                        random_state=100, 
                                                        chunksize=100, 
                                                        passes=5, 
                                                         alpha='symmetric', 
                                                         eta='auto'
                                                        per_word_topics=True)
 
 #       model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=100, update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)
        
        model_list.append(model)
        
        # Compute Coherence score
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values



# Assuming you have coherence_values and x defined
limit = 40
start = 2
step = 3
x = range(start, limit, step)

# Plot results seaborn
sns.lineplot(x=x, y=coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.title("Coherence Score vs. Num Topics")
plt.show()
```

Our analysis revealed an interesting pattern: the coherence metric initially increases up to k=11, then unexpectedly decreases between k=11 and k=17, before rising again to peak at k=33. For simplicity and interpretability, we selected 8 topics, which aligns better with expected news headline subtopics.

![Topic Coherence Score](D:\OneDrive - KU Leuven\5 term\Text mining\project\lda\news_paper_topics\graphs\coherence_topics.png)

*Note: Add your actual coherence graph here*

## Step 5: Applying the Optimal Model to the Full Dataset

After determining the optimal number of topics, we apply the LDA model to the entire dataset:

```python

# Infer topics on the full dataset using the existing LDA model
df['processed_tokens'] = df['headline_text'].apply(preprocess_document)
id2word_full = corpora.Dictionary(df['processed_tokens'])
corpus_full = [id2word_full.doc2bow(text) for text in df['processed_tokens']]


lda_model = gensim.models.ldamulticore.LdaMulticore(workers=6, 
                                                    corpus=corpus_full,
                                                    id2word=id2word_full, 
                                                    num_topics=8,
                                                    random_state=100,
                                                    chunksize=100,
                                                    passes=10, 
                                                    alpha='symmetric', 
                                                    eta='auto',
                                                    per_word_topics=True)

# Visualize the topics
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus_full, id2word_full)
vis

```

## Step 6: Analyzing Topic Evolution Over Time

One of the most insightful aspects of this project is tracking how topics evolve over time:

```python
# Perform inference on the full dataset
topic_distributions_full, _ = lda_model.inference(corpus_full)
df['dominant_topic'] = [np.argmax(topic_dist) for topic_dist in topic_distributions_full]

# Assuming you have a list of manually inputted topic names
manual_topic_names = ["Topic_A", "Topic_B", "Topic_C", "Topic_D", "Topic_E", "Topic_F", "Topic_G", "Topic_H"]

# Assuming df_sample is your DataFrame with the 'dominant_topic' column
# Assign the topic names to the 'topic_name' column in the DataFrame
df['topic_name'] = df['dominant_topic'].map(lambda x: manual_topic_names[x])


# Group by 'year' and 'topic_name' and calculate the count
topic_counts = df.groupby(['year', 'topic_name']).size().reset_index(name='count')

# Plot the evolution of topics over years using seaborn lineplot
plt.figure(figsize=(12, 8))
sns.lineplot(data=topic_counts, x='year', y='count', hue='topic_name')
plt.title("Evolution of Topics Over Years")
plt.xlabel("Year")
plt.ylabel("Number of Documents")
plt.legend(title='Dominant Topic', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
```

Our analysis revealed that Topic G (Topic 7) showed a substantial increase from 2003, reaching its peak in 2012. However, all topics began to decline thereafter, likely due to the decreasing number of headlines in the dataset.

![Topic Evolution](D:\OneDrive - KU Leuven\5 term\Text mining\project\lda\news_paper_topics\graphs\topics_times.png)

*Note: Add your actual topic evolution graph here*

## Step 7: Interpreting the Topics

Understanding what each topic represents is crucial for deriving insights. Here are the top 10 words for our most prominent topic:

**Topic G (Topic 7)**: man, woman, Melbourne, Donald, news, child, court, police, vaccine, car

This topic appears to encompass stories related to legal matters, crime reporting, and public health (notably vaccine-related news).

```python
# Print top words for each topic
pprint(lda_model.print_topics())
doc_lda1 = lda_model[corpus_full]
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

To run the code in this repository, you'll need to install all requirements using:

```
pip install -r requirements.txt
```

## License

This project is licensed under the terms of the LICENSE file included in this repository.
