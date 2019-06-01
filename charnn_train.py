# Ce code est largement inspiré de l'exemple fourni par le framework Keras,
# le dépôt Github de Minesh Mathew et celui de Michael Zhang.

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.utils.data_utils import get_file
import numpy as np
import random
import pickle
import sys
from keras.utils import plot_model

def sample(preds, temperature=1.0):
    # Retourne l'indice correspondant à la probabilité max des événements
    # du tableau preds. 
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature  # Ajout d'une petite constante pour éviter log(0)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def test_model(model, char_to_indices, indices_to_char, seed_string=" ", temperature=1.0, test_length=150):
    """
    L'utilisation d'une température plus élevée permet de créer une séquence
    plus originale que le contenu d'entraînement mais pénalise moins les erreurs
    """
    num_chars = len(char_to_indices.keys())
    for i in range(test_length):
        test_in = np.zeros((1, len(seed_string), num_chars))
        for t, char in enumerate(seed_string):
            test_in[0, t, char_to_indices[char]] = 1
        entire_prediction = model.predict(test_in, verbose=0)[0]
        next_index = sample(entire_prediction[-1], temperature)
        next_char = indices_to_char[next_index]
        seed_string = seed_string + next_char
    return seed_string


# Paramètres
origin = "version_1"
seed = 2

# LSTM
unit_size = 512
num_layers = 3
dropout = 0.4

# optimization parameters
optimizer = 'rmsprop'
training_epochs = 100

maxlen = 200 # Longueur max des phrases
step = 13

# Longueur de la phrase de test à générer
test_length = 240

# Jeu de données
text = open("dataset_fables.txt").read()

# Suppresion des caractères accentués (83 -> 61 caractères)
text = text.replace("œ", "oe")
text = text.replace("’", "'")
text = text.replace("(", "")
text = text.replace(")", "")
text = text.replace("À", "A")
text = text.replace("Â", "A")
text = text.replace("â", "a")
text = text.replace("à", "a")
text = text.replace("ç", "c")
text = text.replace("Ç", "C")
text = text.replace("Ê", "E")
text = text.replace("É", "E")
text = text.replace("è", "e")
text = text.replace("é", "e")
text = text.replace("ê", "e")
text = text.replace("û", "u")
text = text.replace("ù", "u")
text = text.replace("ü", "u")
text = text.replace("î", "i")
text = text.replace("ï", "i")
text = text.replace("ô", "o")
text = text.replace("Ô", "O")


np.random.seed(seed)
random.seed(seed)

print('Longueur du corpus : ', len(text))
chars = sorted(list(set(text)))
num_chars = len(chars)
print('Liste des caractères : ', chars)
print('Nombre de caractères différents :', num_chars)


# Dictionnaires permettant de convertir les caractères en nombre et vice-vera
char_to_indices = dict((c, i) for i, c in enumerate(chars))
indices_to_char = dict((i, c) for i, c in enumerate(chars))
pickle.dump(char_to_indices, open("modeles/{}c2i.p".format(origin), "wb"))
pickle.dump(indices_to_char, open("modeles/{}i2c.p".format(origin), "wb"))

# Découpe le corpus en phrase semi-redondante de longueur maxlen
sentences = []
targets = []
for i in range(0, len(text) - maxlen - 1, step):
    sentences.append(text[i: i + maxlen])
    targets.append(text[i + 1: i + maxlen + 1])
print('Nombre de séquences : ', len(sentences))

print('Vectorisation...')
# La vectorisation permet de ne pas avoir à envoyer directements des nombres
# (correspondant aux lettres) au réseau de neurones. Il ne s'agit pas de
# variables ordinales mais de variables catégorielles. On encode donc
# les nombres en tableau de 0 et de 1. (one hot encoding)
X = np.zeros((len(sentences), maxlen, num_chars), dtype=np.bool)
y = np.zeros((len(sentences), maxlen, num_chars), dtype=np.bool)
for i in range(len(sentences)):
    sentence = sentences[i]
    target = targets[i]
    for j in range(maxlen):
        X[i][j][char_to_indices[sentence[j]]] = 1
        y[i][j][char_to_indices[target[j]]] = 1

print('Construction du modèle...')
model = Sequential()

model.add(LSTM(unit_size, input_dim=num_chars, return_sequences=True))
for i in range(num_layers - 1):
    if dropout:
        model.add(Dropout(dropout))
    model.add(LSTM(unit_size, return_sequences=True))
if dropout:
    model.add(Dropout(dropout))
model.add(TimeDistributed(Dense(num_chars)))
model.add(Activation('softmax'))
model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                )
print(model.summary())
plot_model(model, to_file='model.png')
print('Modèle construit.')

# Enregistrement du texte généré en fichier texte.
outfile = open("sortie/{}.txt".format(origin), "w")

pertes = []

# ----- Entraînement -----
for i in range(training_epochs):
    print('-' * 10 + ' Iteration : {} '.format(i) + '-' * 10)
    outfile.write("\n" + '-' * 10 + ' Iteration : {} '.format(i) + '-' * 10 + "\n")

    # On teste le réseau toutes les 5 itérations
    if i % 5 == 0:
        for temperature in [0.3, 0.6, 1, 1.5, 2]:
            generated_string = test_model(model,
                                        char_to_indices=char_to_indices,
                                        indices_to_char=indices_to_char,
                                        temperature=temperature,
                                        test_length=test_length)
            output = "\n\nTemperature : {}\nFable générée :\n{}".format(temperature, generated_string)
            print(output)
            outfile.write(output + "\n")
            outfile.flush()
            model.save("modeles/{}-{}.h5".format(origin, i))

    history = model.fit(X, y, batch_size=256, epochs=1, verbose=1)
    perte = history.history['loss'][0]
    print('La perte est de {}'.format(perte))
    pertes.append(perte)

outfile.close()
model.save("modeles/{}.h5".format(origin))
print("Evolution de la perte : " + str(pertes))
