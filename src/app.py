from flask import Flask, render_template, request, jsonify
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from flask_cors import CORS
import numpy as np
import pandas as pd
import string
import nltk
nltk.download('stopwords')


DATA_PATH = "data/spam_data.csv"
df = pd.read_csv(DATA_PATH)
category = df["Category"].values.tolist()
messages = df["Message"].values.tolist()


def lowercase(text):
    return text.lower()


def punctuation_removal(text):
    # tạo table để translate thay thế
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)


def tokenizer(text):
    return text.split()


def remove_stopword(tokens):
    stop_words = nltk.corpus.stopwords.words('english')
    return [token for token in tokens if token not in stop_words]


def teamming(tokens):
    teammer = nltk.PorterStemmer()
    return [teammer.stem(token) for token in tokens]


def preprocess_text(text):
    text = lowercase(text)
    text = punctuation_removal(text)
    tokens = tokenizer(text)
    tokens = remove_stopword(tokens)
    tokens = teamming(tokens)

    return tokens


def create_dictionary(messages):
    dictionary = []
    for tokens in messages:
        for token in tokens:
            if token not in dictionary:
                dictionary.append(token)

    return dictionary


def create_feature_BoW(tokens, dictionary):
    features = np.zeros(len(dictionary))

    for token in tokens:
        if token in dictionary:
            features[dictionary.index(token)] += 1

    return features


def predict(text, model, dictionary, label_encoder):
    token = preprocess_text(text)
    features = create_feature_BoW(token, dictionary)
    features = np.array(features).reshape(1, -1)
    result = model.predict(features)
    result = label_encoder.inverse_transform(result)

    return result


tokens = [preprocess_text(message) for message in messages]
dictionary = create_dictionary(tokens)
features = [create_feature_BoW(token, dictionary) for token in tokens]
X = np.array(features)

le = LabelEncoder()
y = le.fit_transform(category)

VAL_SIZE = 0.2
TEST_SIZE = 0.125
SEED = 0

X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                  test_size=VAL_SIZE,
                                                  shuffle=True,
                                                  random_state=SEED)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,
                                                    test_size=TEST_SIZE,
                                                    shuffle=True,
                                                    random_state=SEED)

model = GaussianNB()
model = model.fit(X_train, y_train)


y_val_predict = model.predict(X_val)
y_test_predict = model.predict(X_test)

val_score = accuracy_score(y_val, y_val_predict)
test_score = accuracy_score(y_test, y_test_predict)

print(f'Val accuracy: {val_score}')
print(f'Test accuracy: {test_score}')

x = "I am actually thinking a way of doing something useful"
print(predict(x, model, dictionary, le))


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        email_content = request.form['email_content']
        result = predict(email_content, model, dictionary, le)
        return render_template('index.html', result=result, email_content=email_content)
    return render_template('index.html')


@app.route('/api/classify', methods=['POST'])
def api_classify():
    data = request.json
    email_content = data.get('email_content', '')
    result = predict(email_content, model, dictionary, le)
    return jsonify({'result': result})


if __name__ == '__main__':
    app.run(debug=True, port=8080)
