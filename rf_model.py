import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import classification_report

df = pd.read_csv('collegePlace.csv')

x = df.drop('PlacedOrNot', axis='columns')
x = x.drop('Age', axis='columns')
x = x.drop('Hostel', axis='columns')
y = df['PlacedOrNot']
le = preprocessing.LabelEncoder()
x['Gender'] = le.fit_transform(x['Gender'])
x['Stream'] = le.fit_transform(x['Stream'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)

classify = RandomForestClassifier(n_estimators=10, criterion="entropy")
classify.fit(x_train, y_train)

# Make predictions on the test data
y_pred = classify.predict(x_test)

# Calculate and print accuracy, precision, recall, and F1 score
report = classification_report(y_test, y_pred)
print(report)

# Save the trained model
pickle.dump(classify, open('model.pkl', 'wb'))
