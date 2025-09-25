import tensorflow as tf
from tensorflow import keras
import numpy as np
import mlflow
import mlflow.tensorflow

# --- Variables pour les paramètres ---
EPOCHS = 5
BATCH_SIZE = 128
DROPOUT_RATE = 0.2

# Lancement de la session de suivi MLflow
mlflow.tensorflow.autolog() # Astuce : autolog gère beaucoup de choses pour vous !
with mlflow.start_run() as run:
    print("MLflow Run ID:", run.info.run_id)

    # --- Enregistrement des paramètres ---
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("dropout_rate", DROPOUT_RATE)

    # --- Chargement et préparation des données ---
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)

    # --- Construction du modèle (utiliser les variables définies) ---
    model = keras.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(DROPOUT_RATE),
        keras.layers.Dense(10, activation='softmax')
    ])

    # --- Compilation du modèle ---
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # --- Entraînement du modèle ---
    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1
    )

    # --- Évaluation du modèle ---
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Précision sur les données de test: {test_acc:.4f}")

    # --- Enregistrement des métriques ---
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("test_loss", test_loss)

    # --- Sauvegarde du modèle ---
    model.save("mnist_model.h5")
    print("Modèle sauvegardé sous mnist_model.h5")
    # (L'autologging de MLflow sauvegarde déjà le modèle, mais on garde la sauvegarde locale)