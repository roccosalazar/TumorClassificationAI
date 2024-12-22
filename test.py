print("Test configurazione")
print("Conferma di funzionamento")
print("Test completato con successo")
print("Test completato")


from itertools import product

# Sostituisci <Group_Number> con il numero del tuo gruppo
i = 3

# Definizione delle opzioni
data = [
    "version_1.csv", 
    "version_2.xlsx", 
    "version_3.txt", 
    "version_4.json", 
    "version_5.tsv"
]
tech = [
    "Random Subsampling", 
    "K-fold Cross Validation", 
    "Leave-one-out Cross Validation", 
    "Leave-p-out Cross Validation", 
    "Stratified Cross Validation", 
    "Stratified Shuffle Split", 
    "Bootstrap"
]

# Generazione delle combinazioni e selezione
A, B, C = list(product(data, tech, tech))[i]
print("A:", A, "B:", B, "C:", C)
