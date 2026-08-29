# Biblioteke potrebne za analizu podataka
import numpy as np
import pandas as pd

# Pre-procesuiranje i biblioteke potrebne za Nadgledano učenje
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
# Biblioteke za HASH kodovanje kao i za zamenu oznaka
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder


def preprocess_inputs(df):
    # Funkcija za pre-procesuiranje podataka pomoću Scikit learn bibilioteke
    data_frame = df.copy()

    # Deljenje data frame-a df u X i y
    X = data_frame.drop('Failure_Category_encoded', axis=1)
    y = data_frame['Failure_Category_encoded']

    # Train-test deljenje
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=1)

    # Skaliranje X skupa
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    # Pretvaramo excel file sa anonimizovanim podacima u data frame objekat
    # Popunjavamo prazna mesta nulama i brišemo duplikate u slučaju da postoje
    df = pd.read_excel('Test3_Anonymised.xlsx').fillna(0).drop_duplicates()

    # Kolonu koja sadrži imena klasa koje su nam od interesa pretvaramo u numeričku reprezentaciju
    le = LabelEncoder()
    df['Failure_Category_encoded'] = le.fit_transform(df['Failure Category'])

    # Identifikujemo nenumeričke i ne-time/date podatke i pravimo listu od njih
    non_numeric_cols = df.select_dtypes(exclude=np.number).columns
    non_time_cols = df.select_dtypes(include=['datetime', 'timedelta']).columns

    # Kombinujemo pređašne kolone u jednu listu
    non_numeric_and_time_cols = non_numeric_cols.union(non_time_cols)

    # Sve kolone sa tekstualnim podacima pretvaramo u brojeve
    hasher = ce.HashingEncoder(n_components=10, cols=non_numeric_and_time_cols)
    df_encoded = hasher.fit_transform(df)

    # Pokrećemo fukciju koju smo definisali u redu 16, i delimo naš skup podataka u test i train skupove
    X_train, X_test, y_train, y_test = preprocess_inputs(df_encoded)

    # Promenimo broj slučajnih stabala sa 100 na 1000 jer imamo mali skup podataka.
    # Potrebno je promeniti "class_weight" sa "none" na "balanced" zbog činjenice da su naši podaci izuzetno neuravnoteženi.
    modelRF = RandomForestClassifier(n_estimators=1000, criterion='gini', max_depth=None, min_samples_split=2,
                                     min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt',
                                     max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False,
                                     n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight='balanced',
                                     ccp_alpha=0.0, max_samples=None, monotonic_cst=None)
    # Povećaćemo toleranciju (više računarske snage) jer su naši podaci mali (od 0.0001 na 0.00001).
    # Postavićemo maksimalan broj iteracija na 1500 (umesto 1000).
    modelLSVC = LinearSVC(penalty='l2', loss='squared_hinge', dual='warn', tol=0.00001, C=1.0, multi_class='ovr',
                          fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1500)

    modelRF.fit(X_train, y_train)
    modelLSVC.fit(X_train, y_train)

    predictionsRF = modelRF.predict(X_test)
    predictionsLSVC = modelLSVC.predict(X_test)

    # Evaluiramo modele
    accuracy = modelRF.score(X_test, y_test)
    accuracySVC = modelLSVC.score(X_test, y_test)

    print(f"accuracy SVC: {accuracySVC}/n")
    print(f"this is SVC {classification_report(y_test, predictionsLSVC)}")
    print('/n')
    print(f"this is SVC {confusion_matrix(y_test, predictionsLSVC)}")

    print(f"accuracy: {accuracy}/n")
    print(classification_report(y_test, predictionsRF))
    print('/n')
    print(confusion_matrix(y_test, predictionsRF))