import pandas as pd
from preprocessing import ParserFactory, MissingValuesFactory, FeatureScalerFactory

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
        data = MissingValuesFactory.handle_missing_values(strategy=missing_strategy, data=data)
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
        scaled_data = FeatureScalerFactory.scale_features(strategy=scaling_strategy, data=data, exclude_columns=exclude_columns)
    except Exception as e:
        print(f"Errore durante lo scaling delle feature: {e}. Procedo con i dati non scalati.")
        scaled_data = data

    print("Dati dopo lo scaling delle feature:")
    print(scaled_data.head())

    # A questo punto, il dataset è stato preprocessato.
    # I passaggi successivi includono la scelta tra Hold-Out o K-Fold Cross Validation.

if __name__ == "__main__":
    main()
