# Projet : Conception et Déploiement d'un Modèle Deep Learning

Ce dépôt contient le code pour un projet de Travaux Pratiques sur le cycle de vie complet d'un modèle de Deep Learning, de l'entraînement à la conteneurisation et au déploiement via une API.

## Description des Fichiers

-   `train_model.py`: Script Python pour entraîner un réseau de neurones sur le jeu de données MNIST. Il intègre MLflow pour le suivi des expérimentations et sauvegarde le modèle entraîné sous `mnist_model.h5`.
-   `app.py`: Une application web Flask qui charge le modèle `mnist_model.h5` et expose un endpoint `/predict` pour faire des prédictions.
-   `requirements.txt`: Liste des dépendances Python nécessaires pour faire tourner le projet.
-   `Dockerfile`: Fichier de configuration pour construire une image Docker de l'application Flask, la rendant portable et prête pour le déploiement.
-   `mnist_model.h5`: Le modèle de classification de chiffres Keras/TensorFlow pré-entraîné (généré par `train_model.py`).
-   `mlruns/`: Répertoire généré par MLflow contenant les données des expérimentations. (Il est recommandé d'ajouter ce dossier au `.gitignore`).

## Prérequis

-   Python 3.8+
-   `pip` et `venv`
-   Docker Desktop

## Installation

1.  **Clonez le dépôt :**
    ```bash
    git clone <URL_de_votre_dépôt>
    cd <nom_du_dépôt>
    ```

2.  **Créez et activez un environnement virtuel :**
    ```bash
    python -m venv venv
    # Sur Windows:
    # venv\Scripts\activate
    # Sur macOS/Linux:
    source venv/bin/activate
    ```

3.  **Installez les dépendances :**
    ```bash
    pip install -r requirements.txt
    ```

## Utilisation

### 1. Entraîner le Modèle

Pour entraîner le modèle de classification MNIST, exécutez le script suivant. Cela générera le fichier `mnist_model.h5` et enregistrera l'expérience dans le dossier `mlruns`.

```bash
python train_model.py
```

### 2. Suivre les Expérimentations avec MLflow

Pour visualiser les paramètres et les métriques de vos entraînements, lancez l'interface utilisateur de MLflow :

```bash
mlflow ui
```

Ouvrez votre navigateur et allez à l'adresse `http://127.0.0.1:5000`.

### 3. Lancer l'API via Docker

1.  **Construire l'image Docker :**
    Assurez-vous que Docker Desktop est en cours d'exécution, puis lancez :
    ```bash
    docker build -t mnist-api .
    ```

2.  **Lancer le conteneur :**
    Cette commande démarre l'application et mappe le port 5000 du conteneur au port 5000 de votre machine locale.
    ```bash
    docker run -p 5000:5000 mnist-api
    ```
    L'API est maintenant accessible à l'adresse `http://localhost:5000`.

### 4. Tester l'API

Vous pouvez envoyer une requête POST à l'endpoint `/predict` en utilisant `curl` ou un autre client API. Le payload doit être un JSON contenant une clé `"image"` avec un tableau de 784 pixels (normalisés entre 0 et 1).

**Exemple avec `curl` :**
```bash
# Remplacez le tableau [...] par une vraie image de 784 pixels
curl -X POST -H "Content-Type: application/json" -d "{\"image\": [[0.0, 0.0, ..., 0.9, 0.1, ... 0.0]]}" http://localhost:5000/predict
```