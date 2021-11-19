import nltk
import pandas as pd

nltk.download('brown')

# ------------------------------------------- QUESTION A ------------------------------------------------------ #
from nltk.corpus import brown

# Data by sentences
full_data_sents = brown.tagged_sents(categories=["news"])
train_data_sents = full_data_sents[:int(len(full_data_sents) * 0.9)]
test_data_sents = full_data_sents[int(len(full_data_sents) * 0.9):]

# Data by words
full_data_words = brown.tagged_words(categories=["news"])
train_data_words = full_data_words[:int(len(full_data_words) * 0.9)]
test_data_words = full_data_words[int(len(full_data_words) * 0.9):]

# ------------------------------------------- QUESTION B ------------------------------------------------------ #

probabilities_df = pd.DataFrame(train_data_words, columns=['word', 'tag'])
label_count_df = probabilities_df.groupby(['word', 'tag']).size().reset_index().rename(columns={0: 'count'})
MLE_labels_df = label_count_df.sort_values('count', ascending=False).drop_duplicates(['word'])
MLE_labels_df.drop(['count'], axis=1, inplace=True)


def get_MLE_label(word):
    label = MLE_labels_df[MLE_labels_df['word'] == word]['tag'].values
    label = label[0] if len(label) > 0 else "NN"
    return label


def get_accuracy():
    predictions_df = pd.DataFrame(test_data_words, columns=['word', 'true_label'])
    predictions_df = pd.merge(predictions_df, MLE_labels_df, how="left", on=['word'])
    predictions_df.reset_index(inplace=True, drop=True)
    predictions_df['is_correct'] = predictions_df['tag'] == predictions_df['true_label']
    accuracy = predictions_df['is_correct'].sum() / len(predictions_df.index)
    return (accuracy)
!=
# ------------------------------------------- QUESTION C ------------------------------------------------------ #
