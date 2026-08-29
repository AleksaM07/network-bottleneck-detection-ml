# Data Analysis
import numpy as np
import pandas as pd

# Preprocessing and ML model for Unsupervised learning
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.covariance import EllipticEnvelope, EmpiricalCovariance, GraphicalLasso, LedoitWolf, MinCovDet
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans, SpectralClustering, SpectralCoclustering, SpectralBiclustering
from sklearn.svm import OneClassSVM
from sklearn.linear_model import LogisticRegression
from pyod.models.abod import ABOD
from sklearn.metrics import (accuracy_score, precision_score, classification_report,
                             f1_score, recall_score, silhouette_score, adjusted_rand_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
'''from imblearn.over_sampling import SMOTE  # New import for oversampling
'''

import shap

# For Coding, decoding
import category_encoders as ce


def preprocess_inputs(df):
    # Funkcija za pre-procesuiranje podataka pomoću Scikit learn bibilioteke
    data_frame = df.copy()
    # Deljenje data frame-a df u X i y
    X = data_frame.drop('Limitation', axis=1)
    y = data_frame['Limitation']

    # Train-test deljenje
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1)

    # Skaliranje X skupa
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    shap.initjs()
    # Importing the data that has previously been cleaned
    # Read the first sheet into a DataFrame
    df = pd.read_excel('merged.xlsx').drop_duplicates()
    print(df.shape[0])

    # Since we got all colums with non-numeric data we don't need this anymore
    non_numeric_cols = df.select_dtypes(exclude=np.number).columns
    non_time_cols = df.select_dtypes(include=['datetime', 'timedelta']).columns
    # Combine non-numeric and non-time/date columns
    non_numeric_and_time_cols = non_numeric_cols.union(non_time_cols)

    hasher = ce.HashingEncoder(n_components=10, cols=non_numeric_and_time_cols, return_df=True)
    df_encoded = hasher.fit_transform(df)

    '''# SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    df_encoded, y = smote.fit_resample(df_encoded, df_encoded['Limitation'])
    df_encoded['Limitation'] = y'''

    # Define models
    '''
                                         "Graphical Lasso": GraphicalLasso,
        "Empirical Covariance": EmpiricalCovariance,
        "Ledoit Wolf": LedoitWolf,
        "Min Cov Det": MinCovDet,
        "Spectral Coclustering": SpectralCoclustering(n_clusters=2, random_state=42),
        "Spectral Biclustering": SpectralBiclustering(n_clusters=2, random_state=42),
        # too slow but it might work?
        "Spectral Clustering": SpectralClustering(n_clusters=2, random_state=42)
        "Random Forest": RandomForestClassifier(n_estimators=5000, criterion='gini', max_depth=None,
                                                min_samples_split=2,
                                                min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt',
                                                max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True,
                                                oob_score=False,
                                                n_jobs=None, random_state=None, verbose=0, warm_start=False,
                                                class_weight='balanced',
                                                ccp_alpha=0.0, max_samples=None)'''
    models = {
        "Isolation Forest": IsolationForest(n_estimators=5000, contamination=0.03, max_features=1),
        "Elliptic Envelope1": EllipticEnvelope(contamination=0.03, assume_centered=False),
        "Elliptic Envelope2": EllipticEnvelope(contamination=0.03, assume_centered=True, random_state=42),
        "Elliptic Envelope3": EllipticEnvelope(contamination=0.3, assume_centered=False, random_state=42),
        "Elliptic Envelope4": EllipticEnvelope(contamination=0.3, assume_centered=True, random_state=42),
        "Elliptic Envelope5": EllipticEnvelope(contamination=0.3, support_fraction=1),
        "Local Outlier Factor": LocalOutlierFactor(contamination=0.03),
        "KMeans": KMeans(n_clusters=4),
        #"One Class SVM": OneClassSVM(nu=0.03),
        #"Spectral Clustering": SpectralClustering(n_clusters=2, random_state=42)
        #"Angle-based Outlier Detection50": ABOD(contamination=0.03, method='fast', n_neighbors=10),
        #too slow for now
        #"Angle-based Outlier Detection def 50": ABOD(contamination=0.03, method='default', n_neighbors=50),
        #"Angle-based Outlier Detection 150": ABOD(contamination=0.03, method='fast', n_neighbors=150),
    }

    # Pokrećemo fukciju koju smo definisali, i delimo naš skup podataka u test i train skupove
    X_train, X_test, y_train, y_test = preprocess_inputs(df_encoded)

    predictions = []  # Step 1: Initialize an empty dictionary
    weights = []

    # Train and evaluate each model
    for name, model in models.items():
        print(f"Training and evaluating {name}...")
        if name == "Random Forest":
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        else:
            model.fit(X_train)
            y_pred = model.fit_predict(X_test)

        '''if name in ["KMeans500", "Spectral Clustering"]:
            try:
                # Adjusted Rand Index
                ari = adjusted_rand_score(y_test, y_pred)
                # Silhouette Score
                silhouette = silhouette_score(X_test, y_pred)
                print(f"Adjusted Rand Index: {ari:.4f}, Silhouette Score: {silhouette:.4f}")
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='micro')
                recall = recall_score(y_test, y_pred, average='micro')
                f1 = f1_score(y_test, y_pred, average='micro')

                # Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)
                print(f"Confusion Matrix:\n{cm}")
                #print(f"Classification report: {classification_report(y_test, y_pred, model)}")

                print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
            print()
        
        # Convert outlier detection predictions to binary format if needed
        else:'''
        y_pred = np.where(y_pred == 1, 0, 1)  # Convert from [-1, 1] to [0, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix:\n{cm}")
        #print(f"Classification report: {classification_report(y_test, y_pred, model)}")

        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
        print()

        predictions.append(y_pred)
        weights.append(f1)  # Using F1-score as the weight

    # Convert list of predictions to array
    predictions = np.array(predictions)
    # Majority voting
    combined_predictions = []
    # iterates over the length of the testing labels (y_test), essentially going through each sample in the test set.
    for i in range(len(y_test)):
        weighted_votes = np.zeros(2)  # Assuming binary classification with classes 0 and 1
        for j in range(len(models)):
            weighted_votes[predictions[j, i]] += weights[j]

        combined_predictions.append(np.argmax(weighted_votes))
    '''for i in range(len(y_test)):
        # Check if both models agree or if they predict different classes
        combined_predictions.append(np.argmax(np.bincount(predictions[:, i])))'''

    combined_predictions = np.array(combined_predictions)

    # Evaluate the combined model
    accuracy = accuracy_score(y_test, combined_predictions)
    precision = precision_score(y_test, combined_predictions)
    recall = recall_score(y_test, combined_predictions)
    f1 = f1_score(y_test, combined_predictions)

    cm = confusion_matrix(y_test, combined_predictions)
    print("Combined Model Evaluation:")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
    '''
    explainer = shap.Explainer(iso_forest)
    shap_values = shap.TreeExplainer(iso_forest).shap_values(df_reduced)
    shap.summary_plot(shap_values, df_reduced, plot_type="bar")
    shap.dependence_plot("Subscription Length", shap_values[0], df_reduced, interaction_index="Age")
    '''