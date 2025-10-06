import requests
import json
from PIL import Image
import numpy as np

# URL de l'API qui tourne dans le conteneur Docker
API_URL = "http://localhost:5000/predict"
# Chemin vers l'image que vous avez créée
IMAGE_PATH = "mon_chiffre.png"

def prepare_image(image_path):
    """Charge, pré-traite et prépare une image pour l'API MNIST."""
    # 1. Charger l'image en niveaux de gris
    img = Image.open(image_path).convert('L')
    
    # 2. Redimensionner en 28x28 pixels (format MNIST)
    img = img.resize((28, 28))
    
    # 3. Transformer l'image en tableau NumPy
    image_array = np.array(img)
    
    # 4. Aplatir le tableau 28x28 en un vecteur de 784 éléments
    # et le convertir en liste pour la sérialisation JSON
    return image_array.flatten().tolist()

if __name__ == "__main__":
    # Préparer l'image
    print(f"Préparation de l'image '{IMAGE_PATH}'...")
    image_data = prepare_image(IMAGE_PATH)
    
    # Créer le payload JSON
    payload = json.dumps({"image": image_data})
    
    # Définir les headers
    headers = {'Content-Type': 'application/json'}
    
    print("🚀 Envoi de la requête à l'API...")
    
    try:
        # Envoyer la requête POST à l'API
        response = requests.post(API_URL, headers=headers, data=payload)
        
        # Vérifier si la requête a réussi
        if response.status_code == 200:
            result = response.json()
            prediction = result.get('prediction')
            probabilities = result.get('probabilities', [])
            
            print("\n✅ Prédiction reçue !")
            print(f"   Le modèle pense que ce chiffre est un : {prediction}")
            
            # --- MODIFICATION ICI ---
            if probabilities:
                # Créer une liste de paires (chiffre, probabilité)
                probs_list = probabilities[0]
                predictions_avec_probs = list(enumerate(probs_list))
                
                # Trier la liste par probabilité, en ordre décroissant
                predictions_triees = sorted(predictions_avec_probs, key=lambda item: item[1], reverse=True)
                
                # Afficher les 3 prédictions les plus probables
                print("\n📊 Top des prédictions les plus probables :")
                for i, (chiffre, prob) in enumerate(predictions_triees[:10]):
                    print(f"   {i+1}. Chiffre {chiffre} (Confiance : {prob:.2%})")

        else:
            print(f"\n❌ Erreur: Le serveur a répondu avec le code {response.status_code}")
            print(f"   Message: {response.text}")
            
    except requests.exceptions.ConnectionError as e:
        print("\n❌ Erreur de connexion !")
        print("   Impossible de se connecter à l'API. Assurez-vous que le conteneur Docker est bien en cours d'exécution.")
        print(f"   Commande à lancer : docker run -d -p 5000:5000 mnist-api")