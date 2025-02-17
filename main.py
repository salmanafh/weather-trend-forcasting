#!/usr/bin/env python
# # Weather Trend Forcasting

# Import Data and Packages
import shutil
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import sklearn
import geopandas as gpd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from xgboost import XGBClassifier
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model


df = pd.read_csv("data/482/GlobalWeatherRepository.csv")

different_standards = ["wind_mph", "temperature_fahrenheit", "last_updated_epoch", "pressure_mb", "precip_mm", "feels_like_fahrenheit", "visibility_miles", "gust_mph"]
df = df.drop(columns=different_standards)

weathers = df["condition_text"].unique()


# Data Exploration
numerical_columns = df.select_dtypes(include=[np.number]).columns
categorical_columns = df.select_dtypes(include=[object]).columns
hour_columns = ["sunrise", "sunset", "moonrise", "moonset"]
exception_columns = ["last_updated", "condition_text"]

# Ensure the columns exist before dropping them
columns_to_drop = [col for col in hour_columns + exception_columns if col in categorical_columns]
categorical_columns = categorical_columns.drop(columns_to_drop)
print(df["last_updated"].head())

# Convert the last_updated column to datetime
df["last_updated"] = pd.to_datetime(df["last_updated"])
df["last_updated"].head()

# change hour columns to datetime datatype
for col in hour_columns:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# remove the date in the hour columns
for col in hour_columns:
    df[col] = df[col].dt.time

df.drop(hour_columns, axis=1, inplace=True)

# one-hot encode the categorical columns
encoder = OneHotEncoder()
encoded = encoder.fit_transform(df[categorical_columns])
encoded_df = pd.DataFrame(encoded.toarray(), columns=encoder.get_feature_names_out(categorical_columns))

# concatenate one-hot encoded columns with the original dataframe
df = pd.concat([df, encoded_df], axis=1)
df = df.drop(categorical_columns, axis=1)
df.head()


# Encode the label column
Label_encoder = OneHotEncoder()
encoded_label = Label_encoder.fit_transform(df["condition_text"].values.reshape(-1, 1))
encoded_label

# find outliers of numerical columns with iqrs
def find_outliers_iqr(data):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return data[(data > upper_bound) | (data < lower_bound)]

columns_with_outliers = []
for col in numerical_columns:
    outliers = find_outliers_iqr(df[col])
    if not outliers.empty:
        columns_with_outliers.append(col)
        print(f"Outliers in {col}: {outliers} \n")

# Plot example of outliers before removing
columns_with_outliers[:3]
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
sns.boxplot(x=df[columns_with_outliers[0]], ax=ax[0])
sns.boxplot(x=df[columns_with_outliers[1]], ax=ax[1])
sns.boxplot(x=df[columns_with_outliers[2]], ax=ax[2])
plt.show()

# change outliers with iqr method
def change_outliers_iqr(data):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    data = np.where(data > upper_bound, upper_bound, data)
    data = np.where(data < lower_bound, lower_bound, data)
    return data

for col in numerical_columns:
    # if columns is precip_in skip the change_outliers_iqr function
    if col == "precip_in":
        continue
    df[col] = change_outliers_iqr(df[col])

# check if outliers are changed
for col in numerical_columns:
    outliers = find_outliers_iqr(df[col])
    if not outliers.empty:
        print(f"Outliers in {col}: {outliers} \n")

# plot example of outliers after changing
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
sns.boxplot(x=df[columns_with_outliers[0]], ax=ax[0])
sns.boxplot(x=df[columns_with_outliers[1]], ax=ax[1])
sns.boxplot(x=df[columns_with_outliers[2]], ax=ax[2])
plt.show()

# plot data before normalization
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
sns.histplot(df[numerical_columns[0]], ax=ax[0], kde=True)
sns.histplot(df[numerical_columns[1]], ax=ax[1], kde=True)
sns.histplot(df[numerical_columns[2]], ax=ax[2], kde=True)
plt.show()

# normalize numerical columns
df[numerical_columns] = preprocessing.normalize(df[numerical_columns])

# plot normalized data
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
sns.histplot(df[numerical_columns[0]], ax=ax[0])
sns.histplot(df[numerical_columns[1]], ax=ax[1])
sns.histplot(df[numerical_columns[2]], ax=ax[2])
plt.show()

# find trend in windowed data
windowed_data = df["last_updated"].dt.to_period("M").astype(str)

# plot wind speed every month
plt.figure(figsize=(15, 5))
sns.lineplot(x=windowed_data, y=df["wind_kph"])
plt.title("Wind speed every month")
plt.show()

# plot temperature every month
plt.figure(figsize=(15, 5))
sns.lineplot(x=windowed_data, y=df["temperature_celsius"])
plt.title("Temperature every month")
plt.show()

# plot humidity every month
plt.figure(figsize=(15, 5))
sns.lineplot(x=windowed_data, y=df["humidity"])
plt.title("Humidity every month")
plt.show()

# find correlation between numerical columns
correlation = df[numerical_columns].corr()

# plot correlation of 15 columns
plt.figure(figsize=(15, 10))
sns.heatmap(correlation.iloc[:15, :15], annot=True, fmt=".2f", cmap="coolwarm")
plt.show()

# train test split
X = df.drop(["condition_text", "last_updated"], axis=1)
y = encoded_label.toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

validation_score = []

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    
    validation_score.append({
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
    })


# Gradien Model
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
evaluate_model(xgb, X_test, y_test)

# Ensemble Based Model
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
evaluate_model(rf, X_test, y_test)

# Neural Network Based Model
X_train = tf.data.Dataset.from_tensor_slices(X_train)
y_train = tf.data.Dataset.from_tensor_slices(y_train)

train_dataset = tf.data.Dataset.zip((X_train, y_train))

X_test = tf.data.Dataset.from_tensor_slices(X_test)
y_test = tf.data.Dataset.from_tensor_slices(y_test)

test_dataset = tf.data.Dataset.zip((X_test, y_test))

# shuffle and batch the dataset
train_dataset = train_dataset.shuffle(1024).batch(32)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.batch(32)
test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

# Correct input shape for 2D data
input_layer = Input(shape=(694,))  # Remove the extra None dimension
dense1 = Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001))(input_layer)
dropout1 = Dropout(0.3)(dense1)
dense2 = Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001))(dropout1)
dropout2 = Dropout(0.3)(dense2)
dense3 = Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001))(dropout2)
dropout3 = Dropout(0.3)(dense3)

residual = Dense(64, activation="relu")(input_layer)
residual_connection = tf.keras.layers.add([dropout3, residual])
output_layer = Dense(44, activation="softmax")(residual_connection)

model = Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss="categorical_crossentropy", 
              metrics=["accuracy"])

model.fit(train_dataset,
          epochs=50,
          validation_data=test_dataset)

test_features = test_dataset.map(lambda x, y: x)
test_labels = test_dataset.map(lambda x, y: y)

y_pred = model.predict(test_features)

# Convert test_labels to a NumPy array
test_labels_np = np.concatenate(list(test_labels.as_numpy_iterator()), axis=0)

# Convert y_pred to label indices
y_pred_labels = np.argmax(y_pred, axis=1)

# Convert test_labels_np to label indices
test_labels_np_labels = np.argmax(test_labels_np, axis=1)

accuracy = accuracy_score(test_labels_np_labels, y_pred_labels)
f1 = f1_score(test_labels_np_labels, y_pred_labels, average="weighted")
precision = precision_score(test_labels_np_labels, y_pred_labels, average="weighted")
recall = recall_score(test_labels_np_labels, y_pred_labels, average="weighted")

validation_score.append({
    "accuracy": accuracy,
    "f1": f1,
    "precision": precision,
    "recall": recall
})

validation_score_df = pd.DataFrame(validation_score, index=["XGB", "RandomForest", "NeuralNetwork"])

# plot the validation scores
plt.figure(figsize=(10, 5))
sns.barplot(data=validation_score_df.reset_index(), x="index", y="accuracy")
plt.title("Validation Accuracy")
plt.xlabel("Model")
plt.show()

# Plot the F1 scores
plt.figure(figsize=(10, 5))
sns.barplot(data=validation_score_df.reset_index(), x="index", y="f1")
plt.title("Validation F1 Score")
plt.xlabel("Model")
plt.show()

# Plot the precision scores
plt.figure(figsize=(10, 5))
sns.barplot(data=validation_score_df.reset_index(), x="index", y="precision")
plt.title("Validation Precision Score")
plt.xlabel("Model")
plt.show()

# Plot the recall scores
plt.figure(figsize=(10, 5))
sns.barplot(data=validation_score_df.reset_index(), x="index", y="recall")
plt.title("Validation Recall Score")
plt.xlabel("Model")
plt.show()

# create a new dataframe with the columns that will be used for the model
new_data = pd.DataFrame({
    "country": ["Indonesia"],
    "location_name": ["Jakarta"],
    "latitude": [-6.21],
    "longitude": [106.85],
    "timezone": ["Asia/Jakarta"],
    "temperature_celsius": [20],
    "wind_kph": [10],
    "wind_degree": [90],
    "wind_direction": ["E"],
    "pressure_in": [30],
    "precip_in": [0],
    "humidity": [50],
    "cloud": [30],
    "feels_like_celsius": [20],
    "visibility_km": [10],
    "uv_index": [5],
    "gust_kph": [15],
    "air_quality_Carbon_Monoxide": [0.1],
    "air_quality_Ozone": [0.1],
    "air_quality_Nitrogen_dioxide": [0.1],
    "air_quality_Sulphur_dioxide": [0.1],
    "air_quality_PM2.5": [0.1],
    "air_quality_PM10": [0.1],
    "air_quality_us-epa-index": [1],
    "air_quality_gb-defra-index": [1],
    "moon_phase": ["New Moon"],
    "moon_illumination": [0.1]
})

def do_inference(data):
    # one-hot encode the categorical columns
    encoded = encoder.transform(data[categorical_columns])
    encoded_df = pd.DataFrame(encoded.toarray(), columns=encoder.get_feature_names_out(categorical_columns))

    # concatenate one-hot encoded columns with the original dataframe
    data = pd.concat([data, encoded_df], axis=1)
    data = data.drop(categorical_columns, axis=1)

    # normalize numerical columns
    data[numerical_columns] = preprocessing.normalize(data[numerical_columns])

    # Ensure the data has the correct shape
    data = data.reindex(columns=X.columns, fill_value=0)

    # make inference
    prediction = model.predict(data)
    prediction = np.argmax(prediction, axis=1)
    prediction = weathers[prediction]
    return prediction

do_inference(new_data)

# Geospasial Anaysis of air quality
plt.figure(figsize=(20, 20))
world = gpd.read_file("data/shp/1/ne_110m_admin_0_countries.shp")
ax = plt.gca()
world.plot(ax=ax, color='white', edgecolor='black')

# Create a dictionary to map air quality index values to names
air_quality_names = {
    1: 'Good',
    2: 'Moderate',
    3: 'Unhealthy for Sensitive Groups',
    4: 'Unhealthy',
    5: 'Very Unhealthy',
    6: 'Hazardous'
}

# Map the air quality index values to names
df['air_quality_name'] = df['air_quality_us-epa-index'].map(air_quality_names)

sns.scatterplot(data=df, x="longitude", y="latitude", hue="air_quality_name", palette="coolwarm", ax=ax)
plt.show()

# plot world map and color the countries based on the temperature
plt.figure(figsize=(20, 20))
world = gpd.read_file("data/shp/1/ne_110m_admin_0_countries.shp")
ax = plt.gca()
world.plot(ax=ax, color='white', edgecolor='black')

sns.scatterplot(data=df, x="longitude", y="latitude", hue="temperature_celsius", palette="coolwarm", ax=ax)
plt.show()

