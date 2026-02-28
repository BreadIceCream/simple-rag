import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import jieba


# bm25预处理函数
def nltk_resource_download():
    """Download NLTK resources"""
    print("DOWNLOAD NLTK: Downloading NLTK resources...")
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")


stop_words = set(stopwords.words('english') + stopwords.words('chinese'))
punctuation = set(string.punctuation) | set("，。！？【】（）《》""''：；、—…—")
lemmatizer = WordNetLemmatizer()


def is_english_word(s: str) -> bool:
    return s and 'a' <= s[0].lower() <= 'z'


def bilingual_preprocess_func(text: str) -> list[str]:
    """
    A powerful preprocessing function that supports mixed Chinese and English scenarios
    """
    # 1. transfer to lowercase(typically for English)
    text = text.lower()
    # 2. use jieba to cut text(useful for both Chinese and English)
    tokens = jieba.lcut(text, cut_all=False)
    processed_tokens = []
    for token in tokens:
        # 3. filter stop words and punctuation
        if token in stop_words or token in punctuation:
            continue
        # 4. lemmatize English words
        if is_english_word(token):
            token = lemmatizer.lemmatize(token)
        # 5. filter single character(mostly noises)
        if len(token) > 1:
            processed_tokens.append(token)
    return processed_tokens
