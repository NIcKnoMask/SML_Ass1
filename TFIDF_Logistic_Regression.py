import pandas as pd
from sklearn.linear_model import LogisticRegression
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from random import sample


class Tfidf_Data_Set():

    def __init__(self, paths, human_length, machine_length):
        self.paths = paths
        self.human_length = human_length
        self.machine_length = machine_length

    def load_json(self, path):
        with open(path, 'r') as f:
            json_data = json.load(f)
        return json_data

    def data_processor(self, data):
        processed_data = []
        for i in range(len(data)):
            instance = " ".join(str(x) for x in data[i]['prompt'] + data[i]['txt'])
            processed_data.append(instance)
        return processed_data

    def combine_sets(h1, m1, h2, m2, h1_len, m2_len):
        combined = sample(human_data1, h1_len) + human_data2 + machine_data1 + sample(machine_data2, m2_len)

    def label_generator(self, human_len, machine_len):
        human_labels = [1] * human_len
        machine_labels = [0] * machine_len
        return human_labels + machine_labels

    def veterize(self, data):
        vectorizer = TfidfVectorizer(max_features=5000)
        X = pd.DataFrame(vectorizer.fit_transform(data).todense())
        X.columns = sorted(vectorizer.vocabulary_)
        return X


paths = ["data/set1_human.json", "data/set1_machine.json", "data/set2_human.json", "data/set2_machine.json",
         "data/test.json"]

tfidf = Tfidf_Data_Set(paths, 3600, 3600)

# load json files
human_json1 = tfidf.load_json(tfidf.paths[0])
machine_json1 = tfidf.load_json(tfidf.paths[1])
human_json2 = tfidf.load_json(tfidf.paths[2])
machine_json2 = tfidf.load_json(tfidf.paths[3])

# process the data
# hd1 -> 122584
# md1 -> 3500
# hd2 -> 100
# md2 -> 400
human_data1 = tfidf.data_processor(human_json1)
machine_data1 = tfidf.data_processor(machine_json1)
human_data2 = tfidf.data_processor(human_json2)
machine_data2 = tfidf.data_processor(machine_json2)

# combine four sets
X = tfidf.combine_sets(human_data1, machine_data1, human_data2, machine_data2, 3500, 100)
print(len(X))

# load test set
test_json = tfidf.data_loder(tfidf.paths[4])
test_data = tfidf.data_processor(test_json)
X = X + test_data

# generate labels
labels = tfidf.label_generator(tfidf.human_length, tfidf.machine_length)
print(labels[3590:3610])

# veterize the data
X_df = tfidf.veterize(X)
X_train, X_test = X_df.iloc[:len(labels)], X_df.iloc[len(labels):]

# train data
model = LogisticRegression()
model.fit(X_train, labels)

y_pred = model.predict(X_test)
d = {'Id': [i for i in range(len(y_pred))], 'Predicted': y_pred}
output = pd.DataFrame(d)
output.to_csv('result_logisticRegression.csv', index=False)
