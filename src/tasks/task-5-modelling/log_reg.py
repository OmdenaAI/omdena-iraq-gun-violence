import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sn

# read dataset (downloaded from https://www.kaggle.com/vkrahul/twitter-hate-speech)
df = pd.read_csv('../../data/train_E6oV3lV.csv')
print(df.head())

df.drop_duplicates(inplace=True)

# remove @ from tweets
clean_tweet_clm = []
for _, row in df.iterrows():
    s = row['tweet']
    s_list = s.split()
    tweet_list = []
    for i in range(len(s_list)):
        temp_str = s_list[i]
        if not temp_str.startswith("@"):
            tweet_list.append(temp_str)

    clean_tweet_clm.append(' '.join(tweet_list))

df['clean_tweet'] = clean_tweet_clm
pd.set_option('display.max_columns', None)
print(df.head())

# remove stopwords
stop_words = set(stopwords.words('english'))
df['clean_tweet'] = df['clean_tweet'].apply(
    lambda tweet: ' '.join([word for word in tweet.split() if not word in stop_words]))

# lemmatization
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
df['clean_tweet'] = df['clean_tweet'].apply(
    lambda tweet: ' '.join([lemmatizer.lemmatize(word) for word in tweet.split()]))

# stem
ps = PorterStemmer()
df['clean_tweet'] = df['clean_tweet'].apply(
    lambda tweet: ' '.join([ps.stem(word) for word in tweet.split()]))

# remove hashtag
df['clean_tweet'] = df['clean_tweet'].apply(
    lambda tweet: ' '.join([word.replace('#','') if word.startswith('#') else word for word in tweet.split() ]))

# handle some NaN values
df.replace("", float("NaN"), inplace=True)
df.dropna(subset=['clean_tweet'], inplace=True)
is_NaN = df.isnull()
row_has_NaN = is_NaN.any(axis=1)
rows_with_NaN = df[row_has_NaN]

#save to file
df.to_csv('../../data/twitter_dataset_after_cleaning.csv')

print("---end of cleaning---")

df = pd.read_csv('../../data/twitter_dataset_after_cleaning.csv')

df = df.iloc[:,[2,4]]

df_x = df.drop('label', axis='columns')

X_train, X_test, Y_train, Y_test = train_test_split(df_x, list(df.label), test_size=0.1)

X_train['label'] = Y_train
df = X_train

#downsample
no_hate_df = df[df['label'] == 0]
hate_df = df[df['label'] == 1]

no_hate_df = no_hate_df.sample(n=hate_df.shape[0])

# create train dataset with balanced labels count
dataset_train = pd.concat([hate_df, no_hate_df], axis='rows')
print("train shape: ", dataset_train.shape)

# concat train and test dataset
dataset_test = X_test
dataset_test['label'] = Y_test
print("test shape: ", dataset_test.shape)

concat_ds = pd.concat([dataset_train, dataset_test], axis="rows")

# create corpus, ie, gather all documents in a list
corpus = []
corpus_len = concat_ds.shape[0]
for i in range(corpus_len):
    corpus.append(concat_ds.iloc[i][0])

# TF IDF
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(corpus)
names = tfidf_vectorizer.get_feature_names()
dense = X.todense()
list_dense = dense.tolist()
tfidf_df_temp = pd.DataFrame(list_dense, columns=names)

tfidf_df = tfidf_df_temp

tfidf_df['label_tfidf'] = list(concat_ds['label'])

tfidf_hate = tfidf_df[tfidf_df['label_tfidf']==1]
tfidf_no_hate = tfidf_df[tfidf_df['label_tfidf']==0]
#tfidf_nan = tfidf_df[tfidf_df['label_tfidf'].isna()]

#TODO avoid magic number below (2000). Use %
tfidf_no_hate_train = tfidf_no_hate[:2000]
tfidf_no_hate_test = tfidf_no_hate[2001:]
tfidf_hate_train = tfidf_hate[:2000]
tfidf_hate_test = tfidf_hate[2001:]

dataset_train = pd.concat([tfidf_no_hate_train, tfidf_hate_train], axis ='rows')
dataset_test = pd.concat([tfidf_no_hate_test, tfidf_hate_test], axis ='rows')

X_train = dataset_train.drop(['label_tfidf'], axis=1)
Y_train = list(dataset_train['label_tfidf'])
X_test = dataset_test.drop(['label_tfidf'], axis=1)
Y_test = list(dataset_test['label_tfidf'])

log_reg_model = LogisticRegression()
log_reg_model.fit(X_train, Y_train)
predict = log_reg_model.predict(X_test)
print("acc: ", accuracy_score(Y_test, predict))

# confusion matrix
conf_matrix = confusion_matrix(Y_test, predict)
print("confusion: ", conf_matrix)
df_confusion = pd.DataFrame(conf_matrix, [0,1], columns=[0,1])
plt.figure(figsize=(10,7))
sn.heatmap(df_confusion, annot=True)
plt.show()


print("END2")
