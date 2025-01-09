import numpy as np

# CASO LEAVE P OUT
data = [
    # Reali,    previste
    [[1, 2, 1], [1, 1, 1]],  # Prima coppia
    [[1, 1, 1], [1, 1, 1]],  # Seconda coppia
    [[1, 2, 1], [1, 1, 1]],  # Terza coppia
    [[1, 1, 1], [1, 1, 1]],  # Quarta coppia
    # Aggiungi altre coppie
]

# CASO HOLDOUT
data = [
    [[1, 2, 1], [1, 1, 1]],
]

# PUNTO DI INGRESSO DELLE METRICHE:
predizioni = np.array(data)

lista = [1, 2, 3, 4]

for elementi in lista:
    print(f"Fai i calcoli per coppia {elementi}-esima...")

"""

# Accesso alle previsioni e alle etichette reali
etichette_previste = predizioni[:, 0, :]
print(f"Etichette previste: {etichette_previste}")
etichette_reali = predizioni[:, 1, :]
print(f"Etichette reali: {etichette_reali}")
# Calcolo dell'accuratezza
accuratezza = np.mean(etichette_previste == etichette_reali)
print(f"Accuratezza: {accuratezza}")
"""
