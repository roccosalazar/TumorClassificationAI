from abc import ABC, abstractmethod
import pandas as pd
from preprocessing import ParserFactory, MissingValuesStrategyManager, FeatureScalerStrategyManager
from validation import Holdout, KFold  # Importa le classi implementate

def main():
    # Step 1: Input dell'utente per il percorso del file
    file_path = input("Inserisci il percorso del file del dataset che vuoi analizzare: ").strip()
    if not file_path:
        print("Percorso non fornito. Utilizzo del file di default: 'data/version_1.csv'")
        file_path = 'data/version_1.csv'

    print("Parsing del dataset in corso...")
    try:
        parser = ParserFactory.get_parser(file_path)
        data = parser.parse(file_path)
    except Exception as e:
        print(f"Errore durante il parsing del file: {e}. Utilizzando un dataset vuoto per default.")
        data = pd.DataFrame()

    if data.empty:
        print("Il dataset è vuoto. Termino l'esecuzione.")
        return

    print("Dati originali:")
    print(data.head())

    # Step 2: Scelta dell'utente per la gestione dei valori mancanti
    print("Come vuoi gestire i valori mancanti?")
    print("Opzioni: remove | mean | median | mode")
    missing_strategy = input("Inserisci la tua scelta: ").strip().lower()

    if missing_strategy not in ['remove', 'mean', 'median', 'mode']:
        print("Scelta non valida. Verrà utilizzata la strategia 'median' per default.")
        missing_strategy = 'median'

    print(f"Gestione dei valori mancanti con la strategia: {missing_strategy}...")
    try:
        data = MissingValuesStrategyManager.handle_missing_values(strategy=missing_strategy, data=data)
    except Exception as e:
        print(f"Errore durante la gestione dei valori mancanti: {e}. Procedo con i dati originali.")

    print("Dati dopo la gestione dei valori mancanti:")
    print(data.head())

    # Step 3: Scelta dell'utente per il Feature Scaling
    print("Come vuoi scalare le feature?")
    print("Opzioni: normalize | standardize")
    scaling_strategy = input("Inserisci la tua scelta: ").strip().lower()

    if scaling_strategy not in ['normalize', 'standardize']:
        print("Scelta non valida. Verrà utilizzata la strategia 'normalize' per default.")
        scaling_strategy = 'normalize'

    exclude_columns = ['Sample code number', 'classtype_v1']
    print(f"Applicazione dello scaling delle feature con la strategia: {scaling_strategy}...")
    try:
        scaled_data = FeatureScalerStrategyManager.scale_features(strategy=scaling_strategy, data=data, exclude_columns=exclude_columns)
    except Exception as e:
        print(f"Errore durante lo scaling delle feature: {e}. Procedo con i dati non scalati.")
        scaled_data = data

    print("Dati dopo lo scaling delle feature:")
    print(scaled_data.head())

    # Step 4: Scelta della Validazione
    print("\nScegli il metodo di validazione:")
    print("1. Holdout")
    print("2. K-Fold")
    validation_choice = input("Inserisci il numero del metodo di validazione: ").strip()

    if validation_choice == "1":
        # Configurazione Holdout
        print("\nConfigura la validazione Holdout:")
        test_size = float(input("Inserisci la proporzione per il test set (tra 0.1 e 0.5, default 0.2): ") or 0.2)
        random_state = input("Inserisci il seed per il mescolamento (default nessun seed): ")
        random_state = int(random_state) if random_state else None

        print(f"Configurazione scelta: test_size={test_size}, random_state={random_state}")
        holdout = Holdout(test_size=test_size, random_state=random_state)
        train, test = holdout.split(scaled_data)
        print(f"\nHoldout: Training Set -> {len(train)} righe, Test Set -> {len(test)} righe")

    elif validation_choice == "2":
        # Configurazione K-Fold
        print("\nConfigura la validazione K-Fold:")
        n_splits = int(input("Inserisci il numero di fold (min 2, default 5): ") or 5)
        random_state = input("Inserisci il seed per il mescolamento (default nessun seed): ")
        random_state = int(random_state) if random_state else None

        print(f"Configurazione scelta: K={n_splits}, random_state={random_state}")
        kfold = KFold(n_splits=n_splits, random_state=random_state)
        folds = kfold.split(scaled_data)

        print("\nSuddivisione completata! Ecco le dimensioni di ciascun fold:")
        for i, (train, test) in enumerate(folds):
            print(f"Fold {i + 1}: Training Set -> {len(train)} righe, Test Set -> {len(test)} righe")

    else:
        print("Scelta non valida. Nessuna validazione eseguita.")

if __name__ == "__main__":
    main()  

