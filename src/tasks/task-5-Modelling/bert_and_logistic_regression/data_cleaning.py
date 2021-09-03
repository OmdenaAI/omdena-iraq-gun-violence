import nltk
import pandas as pd

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# read dataset (downloaded from https://www.kaggle.com/vkrahul/twitter-hate-speech)
df = pd.read_csv('../../../data/train_E6oV3lV.csv')
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
df.to_csv('../../../ homedata/twitter_hatespeech_dataset_after_cleaning.csv')

print("---end of cleaning---")