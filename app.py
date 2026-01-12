import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier


st.set_page_config(page_title="ML App - Régression et Classification", layout="centered")
st.title("📊 Application Machine Learning - Régression & Classification")

menu = st.sidebar.radio("📂 Navigation", ["📁 Téléchargement", "📉 Régression", "📊 Classification"])

if menu == "📁 Téléchargement":
    st.header("📁 Téléchargement d'un fichier CSV")
    uploaded_file = st.file_uploader("Téléversez un fichier CSV commun pour la régression et/ou la classification", type=["csv"], key="common_file")

    if uploaded_file:
        with st.spinner("Chargement du fichier..."):
            df = pd.read_csv(uploaded_file)
            st.success("✅ Fichier chargé avec succès")
            st.dataframe(df.head())
            st.session_state["data"] = df

elif menu == "📉 Régression":
    st.header("Régression : Prédire une variable continue")

    if "data" not in st.session_state:
        st.warning("Veuillez d'abord téléverser un fichier CSV dans la section 📁 Téléchargement.")
    else:
        df = st.session_state["data"]
        st.write(df.head())

        target = st.selectbox("Sélectionnez la variable cible (output)", df.columns, key="regression_target")
        if target:
            X = df.drop(columns=[target])
            y = df[target]

            # Prétraitement des colonnes catégorielles
            for col in X.columns:
                if X[col].dtype == object:
                    X[col] = LabelEncoder().fit_transform(X[col])

            # Division des données en ensembles d'entraînement et de test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            models = {
                "Régression Linéaire": LinearRegression(),
                "Arbre de Décision": DecisionTreeRegressor(),
                "Forêt Aléatoire": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor()
            }

            results = []
            with st.spinner("Entraînement des modèles..."):
                for name, model in models.items():
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    results.append({
                        "Modèle": name,
                        "MAE": round(mean_absolute_error(y_test, y_pred), 2),
                        "MSE": round(mean_squared_error(y_test, y_pred), 2),
                        "R²": round(r2_score(y_test, y_pred), 2)
                    })

            st.subheader("📈 Résultats des Modèles")
            st.dataframe(pd.DataFrame(results))

            st.subheader("🔮 Prédiction personnalisée")
            selected_model_name = st.selectbox("Choisir un modèle", list(models.keys()), key="reg_model_choice")
            selected_model = models[selected_model_name]

            user_input = {}
            for col in X.columns:
                val = st.number_input(f"{col}", value=float(df[col].median()), key=f"reg_{col}")
                user_input[col] = val

            if st.button("Prédire (Régression)"):
                input_df = pd.DataFrame([user_input])
                prediction = selected_model.predict(input_df)[0]
                st.success(f"📊 Prédiction estimée : {round(prediction, 2)}")

elif menu == "📊 Classification":
    st.header("Classification : Prédire une catégorie")

    if "data" not in st.session_state:
        st.warning("Veuillez d'abord téléverser un fichier CSV dans la section 📁 Téléchargement.")
    else:
        df = st.session_state["data"]
        st.write(df.head())

        target_clf = st.selectbox("Sélectionnez la variable cible (catégorielle)", df.columns, key="clf_target")
        if target_clf:
            X = df.drop(columns=[target_clf])
            y = df[target_clf]

            # Encodage de la variable cible si nécessaire
            if y.dtype == object or len(np.unique(y)) > 2:
                le = LabelEncoder()
                y = le.fit_transform(y)

            # Encodage des colonnes catégorielles
            for col in X.columns:
                if X[col].dtype == object:
                    X[col] = LabelEncoder().fit_transform(X[col])

            # Division des données en ensembles d'entraînement et de test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            models_clf = {
                "Régression Logistique": LogisticRegression(max_iter=1000),
                "Arbre de Décision": DecisionTreeClassifier(),
                "Forêt Aléatoire": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier()
            }

            results = []
            with st.spinner("Entraînement des modèles..."):
                for name, model in models_clf.items():
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    results.append({
                        "Modèle": name,
                        "Précision": round(accuracy_score(y_test, y_pred), 2),
                        "Rappel": round(recall_score(y_test, y_pred, average='macro'), 2),
                        "F1-score": round(f1_score(y_test, y_pred, average='macro'), 2)
                    })

            st.subheader("📉 Résultats des Modèles")
            st.dataframe(pd.DataFrame(results))

            st.subheader("🔍 Prédiction personnalisée")
            selected_model_name = st.selectbox("Choisir un modèle", list(models_clf.keys()), key="clf_model_choice")
            selected_model = models_clf[selected_model_name]

            user_input = {}
            for i, col in enumerate(X.columns):
                val = st.number_input(f"{col}", value=float(df[col].median()), key=f"clf_{col}_{i}")
                user_input[col] = val

            if st.button("Prédire (Classification)"):
                input_df = pd.DataFrame([user_input])
                prediction = selected_model.predict(input_df)[0]
                st.success(f"🔮 Classe prédite : {prediction}")