import json
import numpy as np
from tensorflow import keras

# Charger une image de test
(_, _), (x_test, _) = keras.datasets.mnist.load_data()
sample_image = x_test[0].reshape(784).tolist() # Prendre la première image, l'aplatir et la convertir en liste

# Créer le payload JSON
payload = {"image": sample_image}

# L'afficher pour le copier-coller dans curl
print(json.dumps(payload))