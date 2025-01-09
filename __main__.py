import pandas as pd
from preprocessing import ParserFactory, MissingValuesStrategyManager, FeatureScalerStrategyManager
from models import KNNClassifier
from validation import Holdout, RandomSubsampling, LeavePOutCV
from metrics import PerformanceMetricsVisualizer

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

        # Separazione delle feature e delle etichette
    labels_column = 'classtype_v1'  # Sostituisci con il nome corretto della colonna delle etichette
    if labels_column not in scaled_data.columns:
        print(f"La colonna delle etichette '{labels_column}' non è presente nel dataset. Termino l'esecuzione.")
        return

    labels = scaled_data[labels_column]
    features = scaled_data.drop(columns=exclude_columns, errors='ignore')

    # Step 4: Scelta della strategia di validazione
    print("Scegli la strategia di validazione:")
    print("1. Holdout")
    print("2. Random Subsampling")
    print("3. Leave-p-Out Cross Validation")
    choice = input("Inserisci la tua scelta (1/2/3): ").strip()

    strategy = None

    try:
        if choice == '1':  # Holdout
            test_size = input("Inserisci la percentuale di test (default 0.2): ").strip()
            test_size = float(test_size) if test_size else 0.2
            strategy = Holdout(test_size=test_size)

        elif choice == '2':  # Random Subsampling
            n_iter = input("Inserisci il numero di iterazioni (default 10): ").strip()
            n_iter = int(n_iter) if n_iter else 10

            test_size = input("Inserisci la percentuale di test (default 0.2): ").strip()
            test_size = float(test_size) if test_size else 0.2

            strategy = RandomSubsampling(n_iter=n_iter, test_size=test_size)

        elif choice == '3':  # Leave-p-Out Cross Validation
            p = input("Inserisci il numero di campioni da lasciare fuori (default 2): ").strip()
            p = int(p) if p else 2

            strategy = LeavePOutCV(p=p)

        else:  # Scelta non valida
            print("Scelta non valida. Uso Holdout come default con percentuale di test: 0.2.")
            strategy = Holdout(test_size=0.2)

    except ValueError as e:
        print(f"Errore nei parametri di validazione: {e}")
        exit()

    # Generazione delle divisioni
    print(f"Generazione delle divisioni utilizzando la strategia: {strategy.__class__.__name__}...")
    validation_data = strategy.generate_splits(features, labels)

        # Mappa i valori di 2 -> 0 (negativo) e 4 -> 1 (positivo)
    mapped_validation_data = [
        (
            [1 if x == 4 else 0 for x in y_real],  # Mappa y_real
            [1 if x == 4 else 0 for x in y_pred]   # Mappa y_pred
        )
        for y_real, y_pred in validation_data  # Applica la trasformazione a ogni coppia
    ]

    # Verifica i dati trasformati
    print("Validation Data Originale:", validation_data)
    print("Validation Data Binaria:", mapped_validation_data)

    # Creazione dell'oggetto PerformanceMetricsVisualizer
    visualizer = PerformanceMetricsVisualizer(mapped_validation_data)

    # Calcolo e visualizzazione delle metriche
    visualizer.visualize_metrics()

    # Salvataggio delle metriche in un file Excel
    visualizer.save("metrics_output.xlsx")

if __name__ == "__main__":
    main()  

