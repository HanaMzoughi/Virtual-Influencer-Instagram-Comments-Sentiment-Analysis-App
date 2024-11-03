from flask import Flask, request, jsonify, send_file, render_template
import re
from io import BytesIO

# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import base64
import io

STOPWORDS = set(stopwords.words("english"))

app = Flask(__name__)


@app.route("/test", methods=["GET"])
def test():
    return "Test request received successfully. Service is running."


@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("landing.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Load the predictor and vectorizer from Models folder
    predictor = pickle.load(open(r"Models/best_svc.pkl", "rb"))
    cv = pickle.load(open(r"Models/tfidf_vectorizer.pkl", "rb"))
    
    try:
        # Check if a file is provided for bulk prediction
        if "file" in request.files:
            # Bulk prediction from CSV file
            file = request.files["file"]
            data = pd.read_csv(file)
            
            predictions, graph = bulk_prediction(predictor, cv, data)

            # Convert the predictions DataFrame to CSV for download
            output = io.StringIO()
            predictions.to_csv(output, index=False)
            output.seek(0)
            
            response = send_file(
                io.BytesIO(output.getvalue().encode()),
                mimetype="text/csv",
                as_attachment=True,
                download_name="Predictions.csv"
            )

            # Encode graph data and add it as a custom header
            response.headers["X-Graph-Exists"] = "true"
            response.headers["X-Graph-Data"] = base64.b64encode(graph.getbuffer()).decode("ascii")

            return response

        # Check if JSON data contains "text" for single prediction
        elif request.is_json and "text" in request.json:
            text_input = request.json["text"]
            predicted_sentiment = single_prediction(predictor, cv, text_input)

            return jsonify({"prediction": predicted_sentiment})

        else:
            return jsonify({"error": "Unsupported Media Type"}), 415

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def single_prediction(predictor , cv, text_input):
    corpus = []
    stemmer = PorterStemmer()
    
    # Preprocess the text input
    review = re.sub("[^a-zA-Z]", " ", text_input)  # Remove non-alphabet characters
    review = review.lower().split()  # Lowercase and split into words
    review = [stemmer.stem(word) for word in review if word not in STOPWORDS]  # Stem words and remove stopwords
    review = " ".join(review)
    corpus.append(review)
    
    # Transform the preprocessed text to TF-IDF features
    X_prediction = cv.transform(corpus)
    
    # Use the predictor (classifier) to predict the class
    y_prediction = predictor.predict(X_prediction)[0]  # Get the single prediction result
    
    # Map the prediction to the corresponding sentiment
    return "Positive" if y_prediction == 2 else "Neutral" if y_prediction == 1 else "Negative"


def bulk_prediction(predictor, cv, data):
    corpus = []
    stemmer = PorterStemmer()

    # Preprocess each sentence
    for i in range(data.shape[0]):
        review = re.sub("[^a-zA-Z]", " ", data.iloc[i]["Sentence"])  # Remove non-alphabet characters
        review = review.lower().split()  # Lowercase and split into words
        review = [stemmer.stem(word) for word in review if word not in STOPWORDS]  # Stem words and remove stopwords
        review = " ".join(review)
        corpus.append(review)

    # Transform the preprocessed corpus to TF-IDF features
    X_prediction = cv.transform(corpus)

    # Use the predictor (classifier) to predict the class for each sentence
    y_predictions = predictor.predict(X_prediction)  # Get predictions for all rows

    # Map numerical predictions to sentiment labels
    y_predictions = list(map(sentiment_mapping, y_predictions))  # Apply mapping function to each prediction

    # Add the predictions to the original data as a new column
    data["Predicted sentiment"] = y_predictions

    # Prepare the CSV output in memory
    predictions_csv = BytesIO()
    data.to_csv(predictions_csv, index=False)
    predictions_csv.seek(0)

    # Generate a graph based on the prediction distribution
    graph = get_distribution_graph(data)

    return predictions_csv, graph

def get_distribution_graph(data):
    fig = plt.figure(figsize=(5, 5))
    colors = ("green", "red")
    wp = {"linewidth": 1, "edgecolor": "black"}
    tags = data["Predicted sentiment"].value_counts()
    explode = (0.01, 0.01)

    tags.plot(
        kind="pie",
        autopct="%1.1f%%",
        shadow=True,
        colors=colors,
        startangle=90,
        wedgeprops=wp,
        explode=explode,
        title="Sentiment Distribution",
        xlabel="",
        ylabel="",
    )

    graph = BytesIO()
    plt.savefig(graph, format="png")
    plt.close()

    return graph


def sentiment_mapping(x):
    if x == 2:
        return "Positive"
    elif x == 1:
        return "Neutral"
    else :
        return "Negative"


if __name__ == "__main__":
    app.run(port=5000, debug=True)