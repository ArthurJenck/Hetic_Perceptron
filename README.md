# Hetic_Perceptron

Implémentation d'un algorithme de Perceptron pour la classification du dataset Iris.

## Installation

Installez les dépendances nécessaires (numpy) avec la commande suivante :

```bash
pip install -r requirements.txt
```

## Utilisation

Lancez le programme pour entraîner le perceptron et afficher les résultats :

```bash
python main.py
```

## Algorithme

- **Architecture** : 3 neurones (one-vs-all)
- **Fonction d'activation** : Heaviside
- **Mise à jour des poids** : `w = w + learning_rate * erreur * exemple`

### Processus d'apprentissage

1. **Initialisation** : Initialiser les poids et le biais
2. **Entraînement** : Pour chaque époque
    - Prendre un échantillon aléatoire
    - Calculer la prédiction
    - Calculer l'erreur
    - Mettre à jour les poids et le biais
3. **Évaluation** : Calculer la précision finale sur le dataset de test

### Résultats

**Précision obtenue** : 90-95%
