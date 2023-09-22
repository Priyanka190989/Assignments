from flask import Flask, render_template, request,redirect, session
import pandas 
from flask_session import Session
import numpy
from scipy.stats import multivariate_normal
from io import BytesIO, StringIO
import matplotlib
from matplotlib import pyplot as plt
import base64
from io import BytesIO
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn 
import os
import math


# Set the backend for matplotlib to generate static images of plots without a GUI
matplotlib.use("Agg")

from stats import (
    no_of_features,
    no_of_numerical_features,
    no_of_qualitative_features,
    find_stats_of_all_numerical_columns,
    predict_gaussian
)

app = Flask(__name__)
app.secret_key = "Testing 123"
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

def get_statistics(data: pandas.DataFrame):
    stats = {}

    stats["no_of_features"] = no_of_features(data)
    stats["no_of_numerical_features"] = no_of_numerical_features(data)
    stats["no_of_qualitative_features"] = no_of_qualitative_features(data)

    stats["stats"] = find_stats_of_all_numerical_columns(data)

    stats["numerical_column_names"] = data.iloc[:,1:].select_dtypes(include=[numpy.number]).columns.values.tolist(),


    # print(stats)

    return stats

@app.route("/")
def home():
    return render_template("index.html")
"""
@app.route("/confusion")
def confusion():
    
    # Render the HTML template with the confusion matrix
    # You can pass the confusion matrix data as a variable to the template
    confusion_matrix_data = calculate_confusion_matrix()  # Define this function to calculate the matrix
    return render_template('confusion.html', matrix_data=confusion_matrix_data)

def calculate_confusion_matrix():
    # Replace these with your actual labels and predicted labels
    
     data_file = request.files["data"].read()
     data_bytes = BytesIO(data_file)
     data = pandas.read_csv(data_bytes)
  
     actual_labels = data['actual_labels'].tolist()
     predicted_labels = data['predicted_labels'].tolist()

    # Calculate the confusion matrix
     matrix = confusion_matrix(actual_labels, predicted_labels)
    
     return matrix


@app.route("/confusion")
def confusion_matrix_plot_encoded(conf_matrix, list_of_classes):

    fig, ax = plt.subplots()

    seaborn.heatmap(conf_matrix, annot=True, xticklabels=list_of_classes, yticklabels=list_of_classes, ax=ax)

    plt.xlabel("Actual")
    plt.ylabel("Predicted")

    temp_file = BytesIO()
    fig.set_figheight(5)
    fig.set_figwidth(5)
    fig.savefig(temp_file, format='png')
    temp_file.seek(0)

    image_encoded = base64.b64encode(temp_file.getvalue()).decode()

    return image_encoded

def image_encodings(data: pandas.DataFrame):

    labels = data.columns.values[0]
    numerical_features = data.iloc[:,1:].select_dtypes(include=[numpy.number]).columns.values

    encodings = []

    for feature in numerical_features:
        encodings.append(plot_encoding(data[[labels, feature]]))

    return encodings

def get_confusion_matrix(data: pandas.DataFrame, split_ratio = 0.2):

    train_test_n = math.floor(data.shape[0] * split_ratio)

    shuffled_data = data.sample(frac=1)

    train_data = shuffled_data.iloc[:-train_test_n]
    test_data = shuffled_data.iloc[-train_test_n:]

    list_of_classes = data.iloc[:, 0].unique().tolist()

    test_x = test_data.iloc[:, 1:].select_dtypes(include=[numpy.number])
    actual_y = test_y = test_data.iloc[:, 0]

    predicted_y = numpy.zeros((test_x.shape[0], 3))

    label_column_name = data.columns.values[0]

    for idx in range(len(list_of_classes)):

        _class = list_of_classes[idx]

        train_x = train_data[train_data[label_column_name] == _class].iloc[:, 1:].select_dtypes(include=[numpy.number])
        train_y = train_data.iloc[:, 0]

        train_mean = train_x.mean()
        train_cov = train_x.cov(ddof=0)

        predicted_y[:, idx] = multivariate_normal.pdf(test_x, train_mean, train_cov)

    actual_y = [list_of_classes.index(row) for row in actual_y]
    predicted_y = numpy.argmax(predicted_y, axis=1).tolist()

    print(actual_y)
    print(predicted_y)
    # print(actual_y, predicted_y)

    conf_matrix = confusion_matrix(actual_y, predicted_y)

    # pyplot.imshow(conf_matrix)


    return conf_matrix, list_of_classes
    # return [] 
   
"""
@app.route("/plot")
def plot():
    data = session['data']
    numeric_columns = data.select_dtypes(include=["number"]).columns
    fig, axes = plt.subplots(
        len(numeric_columns), 1, figsize=(10, 5 * len(numeric_columns))
    )
    for i, feature in enumerate(numeric_columns):
        data.boxplot(column=feature, by=data.columns[0], grid=False, ax=axes[i])
        axes[i].set_title(f"{feature}")

    fig.suptitle("")
    plt.tight_layout()
    # Save the box plots to a BytesIO object
    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    plot_data_uri = base64.b64encode(buffer.read()).decode()
    buffer.close()
   
    return render_template("plot.html", plot_data_uri=plot_data_uri)

@app.route("/result", methods=["GET", "POST"])
def result_page():
    if request.method == "POST":
        data_file = request.files["data"].read()
        data_bytes = BytesIO(data_file)
        data = pandas.read_csv(data_bytes)

        if "data" in session:
            session.pop("data")
        session["data"] = data

    elif "data" in session:
        data = session["data"]
    else:
        return render_template("error_page.html")

    stats = get_statistics(data)
    values = {"sample_data": data.head().to_html(), **stats}

    if "values" in session:
        session.pop("values")

    session["values"] = values

    return render_template("result.html", **values)

@app.route("/class-wise")
def class_wise_distribution():
    if "values" in session:
        return render_template("class_wise_distribution.html", **session["values"])
    else:
        return render_template("error_page.html")

@app.route("/predict", methods = ["GET", "POST"])
def predict():
    values = {**session["values"]}

    # print("values", values)

    if "data" not in session:
        return render_template("error_page.html")

    if request.method == "POST":
        data = session["data"]
        input_x_string = request.form["input"]

        input_x = pandas.read_csv(StringIO(input_x_string), header=None)

        if not input_x.empty:
            results = predict_gaussian(data, input_x)
            values["results"] = results

    return render_template("predict.html", **values)

if __name__ == "__main__":
    app.run(debug=True)
