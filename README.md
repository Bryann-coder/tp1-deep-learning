# Projet de TP : De la conception au déploiement de modèles de Deep Learning

Ce projet couvre le cycle de vie complet d'un modèle de Deep Learning, depuis les concepts fondamentaux jusqu'au déploiement via une API conteneurisée.

## Partie 1 : Concepts Théoriques

### Rappel des modèles linéaires et de l'optimisation stochastique

La **descente de gradient classique (Batch Gradient Descent)** calcule le gradient de la fonction de coût en utilisant **l'intégralité du jeu de données** pour effectuer une seule mise à jour des poids du modèle. C'est précis mais extrêmement lent et gourmand en mémoire pour de grands jeux de données.

La **descente de gradient stochastique (Stochastic Gradient Descent - SGD)** résout ce problème en calculant le gradient et en mettant à jour les poids pour **chaque exemple de données individuellement**.

La **différence principale** réside donc dans la quantité de données utilisées pour chaque mise à jour : tout le dataset pour la GD, un seul exemple pour la SGD.

Dans le contexte du Deep Learning, la **SGD est préférée** car :
1.  **Efficacité sur les grands datasets** : Il est impossible de charger des téraoctets de données en mémoire. La SGD traite les données de manière séquentielle ou par petits lots (mini-batch).
2.  **Convergence plus rapide (en pratique)** : Bien que les mises à jour soient "bruyantes" (variance élevée), elles sont beaucoup plus fréquentes, ce qui permet souvent de converger plus rapidement vers un bon minimum.
3.  **Échapper aux minima locaux** : Le bruit inhérent aux mises à jour de la SGD peut aider l'algorithme à "sauter" hors des minima locaux et à trouver de meilleures solutions.

### Compréhension des réseaux de neurones modernes

Un réseau de neurones est composé de plusieurs couches :
-   **Couche d'entrée (Input Layer)** : C'est la porte d'entrée du réseau. Elle ne fait aucun calcul, elle reçoit simplement les données brutes. Sa taille est égale au nombre de caractéristiques (features) de l'entrée (par exemple, 784 pour une image MNIST de 28x28 pixels).
-   **Couches cachées (Hidden Layers)** : Ce sont les couches intermédiaires entre l'entrée et la sortie. C'est là que la majorité des calculs ont lieu. Chaque neurone d'une couche cachée applique une transformation non-linéaire (via une fonction d'activation comme ReLU) aux données qu'il reçoit de la couche précédente. C'est l'enchaînement de ces couches qui permet au réseau d'apprendre des représentations de plus en plus complexes des données.
-   **Couche de sortie (Output Layer)** : C'est la dernière couche du réseau. Elle produit le résultat final. Sa structure (nombre de neurones et fonction d'activation) dépend de la tâche. Pour une classification à 10 classes, elle aura 10 neurones et une fonction d'activation `softmax`.

Le processus de **rétropropagation du gradient (backpropagation)** est l'algorithme qui permet d'entraîner le réseau. En termes simples :
1.  **Phase "Forward" (passe avant)** : Une donnée d'entrée traverse le réseau de la première à la dernière couche pour produire une prédiction.
2.  **Calcul de l'erreur** : On compare la prédiction à la vraie valeur (l'étiquette) à l'aide d'une fonction de coût (loss function).
3.  **Phase "Backward" (passe arrière)** : L'algorithme calcule la contribution de chaque poids du réseau à l'erreur finale, en propageant l'erreur "à l'envers", de la couche de sortie vers la couche d'entrée, grâce à la règle de dérivation en chaîne.
4.  **Mise à jour des poids** : Les poids de chaque neurone sont ajustés dans la direction qui minimise l'erreur, en utilisant le gradient calculé à l'étape précédente. Ce processus est répété pour de nombreux exemples de données jusqu'à ce que le modèle soit performant.

### Questions sur l'Exercice 1

#### Question 1 : Utilité des couches Dense et Dropout, et de la fonction softmax

-   **Couche `Dense`** : C'est la couche de neurones la plus basique, où chaque neurone est connecté à tous les neurones de la couche précédente. Elle apprend des relations linéaires dans les données qui lui sont présentées, et l'ajout d'une fonction d'activation (comme `relu`) lui permet d'apprendre des relations non-linéaires.
-   **Couche `Dropout`** : C'est une technique de **régularisation** pour lutter contre le **surapprentissage (overfitting)**. Pendant l'entraînement, elle "désactive" aléatoirement une fraction des neurones (ici 20%). Cela force le réseau à apprendre des caractéristiques plus robustes et l'empêche de devenir trop dépendant de quelques neurones spécifiques.
-   **Fonction `softmax`** : Elle est utilisée dans la couche de sortie pour les problèmes de classification multi-classes. Elle transforme les scores bruts (logits) du réseau en une distribution de probabilités, où chaque sortie représente la probabilité que l'entrée appartienne à une classe spécifique. La somme de toutes les probabilités est égale à 1, ce qui rend l'interprétation du résultat très intuitive.

#### Question 2 : L'optimiseur Adam

L'optimiseur **Adam (Adaptive Moment Estimation)** est une amélioration de la SGD simple. Il combine deux concepts clés :
1.  **Momentum** : Il utilise une moyenne mobile des gradients passés pour accélérer la descente dans la bonne direction et amortir les oscillations. C'est comme une balle qui dévale une pente et prend de l'élan.
2.  **Taux d'apprentissage adaptatif (RMSProp)** : Il ajuste le taux d'apprentissage pour chaque poids individuellement, en se basant sur une moyenne mobile des carrés des gradients passés. Cela permet d'effectuer des mises à jour plus grandes pour les poids peu fréquents et plus petites pour les poids fréquents.

En combinant ces deux approches, Adam converge souvent plus vite et de manière plus stable que la SGD simple, nécessitant moins de réglages manuels du taux d'apprentissage.

#### Question 3 : Vectorisation et calculs par lots

-   **Vectorisation** : Ce concept consiste à effectuer des opérations sur des tableaux (vecteurs, matrices) entiers en une seule fois, plutôt que d'itérer sur chaque élément avec une boucle. Les bibliothèques comme NumPy et TensorFlow sont hautement optimisées pour ces opérations. Dans le code, `x_train.astype("float32") / 255.0` est un exemple de vectorisation : la division est appliquée à tous les 60 000 * 784 pixels de la matrice `x_train` simultanément, ce qui est beaucoup plus rapide.
-   **Calculs par lots (Batching)** : Au lieu de traiter les images une par une (SGD pure) ou toutes à la fois (Batch GD), on les traite par "lots" (mini-batches). Ici, `batch_size=128` signifie que le modèle traite 128 images, calcule l'erreur moyenne sur ce lot, puis effectue une seule mise à jour des poids. C'est le compromis parfait : il réduit le bruit de la SGD tout en restant efficace en termes de mémoire et de calcul. `model.fit()` gère ce processus automatiquement.



## Partie 2 : Ingénierie et Déploiement

### Question 1 : Pipeline CI/CD avec GitHub Actions

Un pipeline de CI/CD (Intégration Continue / Déploiement Continu) automatise la construction et le déploiement de notre application à chaque modification du code. Avec GitHub Actions, on peut créer un workflow qui se déclenche, par exemple, à chaque `push` sur la branche `main`.

Voici comment il pourrait fonctionner :

1.  **Déclencheur (Trigger)** : Le pipeline démarre automatiquement lorsqu'un développeur pousse du code sur la branche `main`.

2.  **Étapes du Pipeline (Jobs/Steps)** :
    *   **Build** :
        *   Le service de CI (l'exécuteur de GitHub Actions) récupère la dernière version du code.
        *   Il se connecte à un registre de conteneurs (comme Docker Hub ou Google Artifact Registry).
        *   Il exécute la commande `docker build` pour construire l'image Docker de notre application en utilisant le `Dockerfile`.
        *   Il "tag" l'image avec un identifiant unique (par exemple, l'ID du commit git).
        *   Il pousse (`docker push`) l'image construite vers le registre de conteneurs.
    *   **Deploy** :
        *   Une fois l'image poussée, cette étape se connecte à notre fournisseur de cloud (ex: Google Cloud, AWS).
        *   Elle exécute une commande pour déployer la nouvelle version de l'image sur le service d'hébergement (ex: `gcloud run deploy` pour Google Cloud Run ou des commandes `kubectl` pour Kubernetes).
        *   Le service cloud récupère la nouvelle image depuis le registre et met à jour l'application en cours d'exécution, souvent sans temps d'arrêt (déploiement "rolling update").

Ce processus garantit que chaque nouvelle version du code est automatiquement construite, testée (on pourrait ajouter une étape de test) et déployée, réduisant les erreurs manuelles et accélérant la mise en production.

### Question 2 : Indicateurs clés pour le monitoring en production

Une fois le modèle déployé, il est crucial de le surveiller pour s'assurer qu'il fonctionne correctement et que ses performances ne se dégradent pas. Voici trois types d'indicateurs clés :

1.  **Indicateurs de Performance du Modèle** :
    *   **Dérive de la précision (Accuracy Drift)** : En stockant les prédictions et les vraies étiquettes (si elles deviennent disponibles plus tard), on peut recalculer périodiquement la précision du modèle en production. Une baisse de la précision est un signal fort de "dérive du modèle" (le monde réel a changé).
    *   **Distribution des prédictions** : On surveille la proportion de prédictions pour chaque classe (0, 1, 2...). Si, par exemple, le modèle commence soudainement à prédire "8" pour 90% des requêtes alors qu'il le faisait pour 10% en entraînement, cela peut indiquer un problème avec les données d'entrée ou le modèle lui-même.

2.  **Indicateurs Opérationnels (Santé de l'API)** :
    *   **Latence** : Le temps que met l'API pour retourner une prédiction (en millisecondes). Une augmentation de la latence peut dégrader l'expérience utilisateur. On surveille la moyenne, mais aussi les percentiles (ex: p95, p99) pour détecter les requêtes lentes.
    *   **Taux d'erreur** : Le pourcentage de requêtes qui échouent (codes d'erreur HTTP 5xx). Un taux d'erreur non nul est un signe de problème technique.
    *   **Trafic (Débit)** : Le nombre de requêtes par seconde (RPS). Permet de planifier la mise à l'échelle de l'infrastructure.

3.  **Indicateurs de Dérive des Données (Data Drift)** :
    *   **Distribution des caractéristiques d'entrée** : On surveille les statistiques des données envoyées au modèle (moyenne, écart-type, etc. des pixels de l'image). Si la distribution des données en production s'écarte significativement de celle des données d'entraînement (par exemple, les images deviennent plus sombres), le modèle risque de perdre en performance. C'est un signe précoce que le modèle pourrait avoir besoin d'être ré-entraîné.


    ## Partie 2 : Ingénierie et Déploiement

### Question 1 : Pipeline CI/CD avec GitHub Actions

Un pipeline de CI/CD (Intégration Continue / Déploiement Continu) automatise la construction et le déploiement de notre application à chaque modification du code. Avec GitHub Actions, on peut créer un workflow qui se déclenche, par exemple, à chaque `push` sur la branche `main`.

Voici comment il pourrait fonctionner :

1.  **Déclencheur (Trigger)** : Le pipeline démarre automatiquement lorsqu'un développeur pousse du code sur la branche `main`.

2.  **Étapes du Pipeline (Jobs/Steps)** :
    *   **Build** :
        *   Le service de CI (l'exécuteur de GitHub Actions) récupère la dernière version du code.
        *   Il se connecte à un registre de conteneurs (comme Docker Hub ou Google Artifact Registry).
        *   Il exécute la commande `docker build` pour construire l'image Docker de notre application en utilisant le `Dockerfile`.
        *   Il "tag" l'image avec un identifiant unique (par exemple, l'ID du commit git).
        *   Il pousse (`docker push`) l'image construite vers le registre de conteneurs.
    *   **Deploy** :
        *   Une fois l'image poussée, cette étape se connecte à notre fournisseur de cloud (ex: Google Cloud, AWS).
        *   Elle exécute une commande pour déployer la nouvelle version de l'image sur le service d'hébergement (ex: `gcloud run deploy` pour Google Cloud Run ou des commandes `kubectl` pour Kubernetes).
        *   Le service cloud récupère la nouvelle image depuis le registre et met à jour l'application en cours d'exécution, souvent sans temps d'arrêt (déploiement "rolling update").

Ce processus garantit que chaque nouvelle version du code est automatiquement construite, testée (on pourrait ajouter une étape de test) et déployée, réduisant les erreurs manuelles et accélérant la mise en production.

### Question 2 : Indicateurs clés pour le monitoring en production

Une fois le modèle déployé, il est crucial de le surveiller pour s'assurer qu'il fonctionne correctement et que ses performances ne se dégradent pas. Voici trois types d'indicateurs clés :

1.  **Indicateurs de Performance du Modèle** :
    *   **Dérive de la précision (Accuracy Drift)** : En stockant les prédictions et les vraies étiquettes (si elles deviennent disponibles plus tard), on peut recalculer périodiquement la précision du modèle en production. Une baisse de la précision est un signal fort de "dérive du modèle" (le monde réel a changé).
    *   **Distribution des prédictions** : On surveille la proportion de prédictions pour chaque classe (0, 1, 2...). Si, par exemple, le modèle commence soudainement à prédire "8" pour 90% des requêtes alors qu'il le faisait pour 10% en entraînement, cela peut indiquer un problème avec les données d'entrée ou le modèle lui-même.

2.  **Indicateurs Opérationnels (Santé de l'API)** :
    *   **Latence** : Le temps que met l'API pour retourner une prédiction (en millisecondes). Une augmentation de la latence peut dégrader l'expérience utilisateur. On surveille la moyenne, mais aussi les percentiles (ex: p95, p99) pour détecter les requêtes lentes.
    *   **Taux d'erreur** : Le pourcentage de requêtes qui échouent (codes d'erreur HTTP 5xx). Un taux d'erreur non nul est un signe de problème technique.
    *   **Trafic (Débit)** : Le nombre de requêtes par seconde (RPS). Permet de planifier la mise à l'échelle de l'infrastructure.

3.  **Indicateurs de Dérive des Données (Data Drift)** :
    *   **Distribution des caractéristiques d'entrée** : On surveille les statistiques des données envoyées au modèle (moyenne, écart-type, etc. des pixels de l'image). Si la distribution des données en production s'écarte significativement de celle des données d'entraînement (par exemple, les images deviennent plus sombres), le modèle risque de perdre en performance. C'est un signe précoce que le modèle pourrait avoir besoin d'être ré-entraîné.