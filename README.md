# Projet : Cycle de Vie d'un Modèle de Reconnaissance de Chiffres (MNIST)

Ce projet illustre le cycle de vie complet d'un modèle de Deep Learning, de l'expérimentation à la production. Il comprend l'entraînement d'un classificateur de chiffres sur le jeu de données MNIST, le suivi des expériences avec MLflow, l'encapsulation du meilleur modèle dans une API web Flask, et son déploiement via Docker.

## Architecture du Projet

Le projet est structuré pour séparer clairement les différentes étapes du cycle de vie du modèle : entraînement, service (API) et test.

```
/
|-- mlruns/                     # (Généré) Dossier de MLflow pour le suivi des expériences
|-- mnist_model.h5              # Modèle entraîné, prêt à être utilisé par l'API
|-- train_model.py              # Script pour entraîner et comparer les modèles
|-- app.py                      # Application Flask pour servir le modèle via une API REST
|-- test.py                     # Script client pour tester l'API
|-- Dockerfile                  # Instructions pour construire l'image Docker de l'API
|-- requirements.txt            # Dépendances Python du projet
|-- README.md                   # Ce fichier
`-- .gitignore                  # Fichiers et dossiers à ignorer par Git
```

### Rôle des Fichiers

| Fichier                 | Description                                                                                                                                                                                                |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`train_model.py`**    | **Script d'entraînement** : Charge les données MNIST, définit une architecture de réseau de neurones, et entraîne plusieurs versions du modèle en comparant différents optimiseurs (`Adam`, `SGD`, `RMSprop`). Utilise **MLflow** pour enregistrer les paramètres, les métriques et les modèles de chaque expérimentation. |
| **`app.py`**            | **Application API** : Utilise le framework **Flask** pour créer un serveur web. Il charge le fichier `mnist_model.h5` et expose un endpoint `/predict` qui accepte des données d'image en JSON et retourne une prédiction. |
| **`test.py`**           | **Script de test client** : Simule un client qui interroge l'API. Il charge une image locale (`mon_chiffre.png`), la pré-traite pour qu'elle corresponde au format attendu par le modèle, envoie une requête POST à l'API Dockerisée et affiche la prédiction de manière lisible. |
| **`Dockerfile`**        | **Fichier de conteneurisation** : Contient les instructions pour construire une image **Docker**. Cette image embarque Python, les dépendances nécessaires, le code de l'API (`app.py`) et le modèle (`mnist_model.h5`) pour créer un conteneur portable et isolé. |
| **`requirements.txt`**  | **Dépendances** : Liste toutes les bibliothèques Python nécessaires (`tensorflow`, `flask`, `mlflow`, `numpy`, `requests`, `Pillow`) pour que le projet fonctionne. |
| **`mnist_model.h5`**    | **Le modèle** : Fichier binaire contenant l'architecture et les poids du réseau de neurones entraîné. **Note importante** : ce fichier n'est pas directement généré. Il doit être récupéré manuellement depuis les "artefacts" de la meilleure exécution dans l'interface de MLflow. |
| **`mlruns/`**           | **Suivi MLflow** : Répertoire créé et géré automatiquement par MLflow. Il contient toutes les informations sur les exécutions (paramètres, métriques, artefacts comme les modèles) de manière structurée. |

---

## Guide d'Utilisation Complet

Suivez ces étapes pour exécuter le projet du début à la fin.

### Étape 1 : Prérequis

Assurez-vous d'avoir installé les outils suivants sur votre machine :
-   Python 3.8+ et `pip`
-   Git
-   Docker Desktop (et assurez-vous qu'il est en cours d'exécution)

### Étape 2 : Installation

1.  **Clonez le dépôt et naviguez dans le répertoire :**
    ```bash
    git clone <URL_de_votre_dépôt>
    cd <nom_du_dépôt>
    ```

2.  **Créez un environnement virtuel et activez-le :**
    ```bash
    python -m venv venv
    # Sur Windows
    venv\Scripts\activate
    # Sur macOS/Linux
    source venv/bin/activate
    ```

3.  **Créez le fichier `requirements.txt`** avec le contenu suivant :
    ```txt
    tensorflow
    flask
    numpy
    mlflow
    requests
    Pillow
    ```

4.  **Installez les dépendances :**
    ```bash
    pip install -r requirements.txt
    ```

### Étape 3 : Entraînement et Sélection du Modèle

1.  **Lancez le script d'entraînement :**
    Ce script va entraîner trois modèles différents et enregistrer les résultats dans MLflow.
    ```bash
    python train_model.py
    ```

2.  **Visualisez les résultats avec MLflow :**
    Lancez l'interface utilisateur de MLflow pour comparer les performances des modèles.
    ```bash
    mlflow ui
    ```
    Ouvrez votre navigateur à l'adresse [http://127.0.0.1:5000](http://127.0.0.1:5000).

3.  **Exportez le meilleur modèle :**
    a. Dans l'interface MLflow, trouvez l'exécution (`run`) avec la meilleure `final_test_accuracy`.
    b. Cliquez sur cette exécution pour voir ses détails.
    c. Dans la section **"Artifacts"**, vous verrez un dossier (`model_Adam`, par exemple). Cliquez dessus.
    d. Vous y trouverez le fichier `model.h5`. **Téléchargez-le, placez-le à la racine de votre projet et renommez-le `mnist_model.h5`**.

    Votre projet doit maintenant contenir le fichier `mnist_model.h5`.

### Étape 4 : Déploiement de l'API avec Docker

Maintenant que nous avons le modèle, nous pouvons construire l'image Docker de notre API.

1.  **Construisez l'image Docker :**
    Cette commande lit le `Dockerfile` et assemble une image nommée `mnist-api`.
    ```bash
    docker build -t mnist-api .
    ```

2.  **Lancez le conteneur Docker :**
    Cette commande démarre un conteneur à partir de l'image, en mappant le port 5000 du conteneur au port 5000 de votre machine.
    ```bash
    docker run -p 5000:5000 mnist-api
    ```
    L'API est maintenant en cours d'exécution et accessible à `http://localhost:5000`.

### Étape 5 : Tester l'API de Prédiction

Pour tester l'API, nous allons utiliser le script `test.py`, qui nécessite une image de chiffre.

1.  **Créez une image de test :**
    -   Ouvrez un logiciel de dessin simple (comme Paint, GIMP, etc.).
    -   Créez une petite image (ex: 100x100 pixels) avec un **fond noir**.
    -   Dessinez un chiffre **en blanc** au centre.
    -   Enregistrez l'image sous le nom `mon_chiffre.png` à la racine de votre projet.

2.  **Exécutez le script de test :**
    Assurez-vous que votre conteneur Docker est toujours en cours d'exécution, puis lancez :
    ```bash
    python test.py
    ```

3.  **Analysez le résultat :**
    Le script affichera la prédiction du modèle ainsi que les 10 prédictions les plus probables avec leur score de confiance.

    **Exemple de sortie attendue :**
    ```
    Préparation de l'image 'mon_chiffre.png'...
    🚀 Envoi de la requête à l'API...

    ✅ Prédiction reçue !
       Le modèle pense que ce chiffre est un : 7

    📊 Top des prédictions les plus probables :
       1. Chiffre 7 (Confiance : 99.85%)
       2. Chiffre 2 (Confiance : 0.11%)
       3. Chiffre 1 (Confiance : 0.02%)
       ...
    ```

Félicitations ! Vous avez complété tout le cycle : entraînement, sélection, déploiement et test d'un modèle de Deep Learning.