
# AI per la Classificazione dei Tumori al Seno (Breast Cancer Classification)  
## Descrizione
  ### Contesto  
Le cellule tumorali possono essere classificate in due principali categorie: benigne e maligne. Le cellule benigne sono caratterizzate da una crescita limitata e localizzata, senza la capacità di invadere altri tessuti o diffondersi in altre parti del corpo. Al contrario, le cellule maligne hanno un comportamento aggressivo, con un'elevata capacità di proliferazione, invasione e metastasi. Identificare in modo accurato la natura delle cellule tumorali è cruciale per garantire una diagnosi precoce e un trattamento efficace.
La classificazione delle cellule tumorali si basa su caratteristiche morfologiche e biologiche, quali la forma e la dimensione delle cellule, l'aspetto dei nuclei, il grado di adesione cellulare e il numero di mitosi. Queste informazioni, raccolte attraverso test citologici, forniscono dati fondamentali per distinguere tra cellule benigne e maligne.
L'obiettivo principale di questo progetto è costruire un modello di machine learning in grado di classificare le cellule tumorali in benigne o maligne. Per raggiungere questo scopo, verrà utilizzata una variante del famoso Breat Cancer Winsconsin Dataset [Breast Cancer Wisconsin (Original)](http://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(original)), spesso utilizzato in progetti di classificazione per identificare tumori benigni e maligni. Il modello sarà progettato per supportare il processo decisionale dei medici, aumentando l'accuratezza e la velocità della diagnosi. L'approccio adottato sfrutta diverse strategie di validazione, come il metodo di Holdout, il Random Subsampling e la Leave-p-Out Cross Validation, per garantire una valutazione accurata del modello. Il fulcro dell'analisi è rappresentato dall'uso del classificatore k-NN (k-Nearest Neighbors) per distinguere tra tumori benigni e maligni.
Il programma è progettato per offrire agli utenti un'ampia personalizzazione attraverso una serie di opzioni interattive, che consentono di configurare l'intero processo, dalla gestione dei dati mancanti al tipo di scaling delle feature, fino alla strategia di validazione scelta. I risultati dell'analisi possono essere esplorati in due modalità principali: la generazione di file di output dettagliati in formato Excel e la visualizzazione di grafici riepilogativi delle metriche di performance. Questi strumenti forniscono agli utenti una chiara comprensione delle prestazioni del modello, rappresentando un supporto prezioso per l'analisi diagnostica dei tumori.
## Il Dataset `version_1.csv`
Il dataset `version_1.csv` utilizzato in questo progetto è fondamentale per l'analisi e la classificazione dei tumori. Ecco una panoramica dettagliata del dataset:
- **Numero di Campioni**: 693 campioni.
- **Numero di Caratteristiche**: 13 caratteristiche per campione.
- **Nomi delle Caratteristiche**:
- `Blood Pressure`: Pressione sanguigna registrata (probabilmente un dato aggiuntivo non direttamente correlato alle cellule).
- `Mitoses`: Frequenza delle mitosi, indicativa del grado di proliferazione cellulare.
- `Sample code number`: Identificativo univoco per ogni campione di analisi.
- `Normal Nucleoli`: Numero di nucleoli normali presenti nelle cellule.
- `Single Epithelial Cell Size`: Dimensione della singola cellula epiteliale, un indicatore della regolarità.
- `Uniformity of Cell Size`: Uniformità delle dimensioni cellulari, un valore più alto può indicare malignità.
- `Clump Thickness`: Spessore del gruppo di cellule, usato per valutare la densità dei campioni.
- `Heart Rate`: Frequenza cardiaca (probabilmente aggiunto per scopi di studio, non strettamente legato alla classificazione).
- `Marginal Adhesion`: Capacità delle cellule di aderire tra loro.
- `Bland Chromatin`: Cromatina omogenea, legata all'aspetto dei nuclei delle cellule.
- `classtype_v1`: Classificazione delle cellule tumorali (2 per benigno, 4 per maligno).
- `Uniformity of Cell Shape`: Uniformità della forma delle cellule, importante per identificare alterazioni morfologiche.
- `Bare Nucleix_wrong`: Nuclei scoperti (probabilmente un errore di digitazione, riferito a "Bare Nuclei").
  

### **Anteprima del Dataset**

| Blood Pressure | Mitoses | Sample Code Number | Normal Nucleoli | Single Epithelial Cell Size | Uniformity of Cell Size | Clump Thickness | Heart Rate | Marginal Adhesion | Bland Chromatin | Class Type | Uniformity of Cell Shape | Bare Nucleix |
|----------------|---------|--------------------|-----------------|----------------------------|-------------------------|-----------------|------------|-------------------|-----------------|------------|--------------------------|--------------|
| 95             | 1       | 1000025.0         | 1               | 2.0                        | 1.0                     | 5.0             | 63         | 1.0               | 3.0             | 2.0        | 1.0                      | 1.0          |
| 100            | 1       | 1002945.0         | 2               | 7.0                        | 4.0                     | 5.0             | 66         | 5.0               | 3.0             | 2.0        | 4.0                      | 10.0         |
| 112            | 1       | 1015425.0         | 1               | 2.0                        | NaN                     | NaN             | 72         | 1.0               | 3.0             | NaN        | 1.0                      | 2.0          |
| 99             | 1       | 1016277.0         | 7               | 3.0                        | 8.0                     | 6.0             | 98         | 1.0               | 3.0             | 2.0        | 8.0                      | 4.0          |
| 122            | 1       | 1017023.0         | 1               | 2.0                        | 1.0                     | 4.0             | 66         | 3.0               | 3.0             | 2.0        | 1.0                      | 1.0          |

# **File Principale: main.py**
## **Descrizione Generale**
Il file `main.py` è il cuore del progetto. Coordina l'intero flusso di lavoro, dalla lettura del dataset all'elaborazione dei dati, dalla classificazione alla visualizzazione dei risultati. Questo script è progettato per essere interattivo e offre molteplici opzioni configurabili.
## Come Eseguire il Codice
Per eseguire il codice di questo progetto, seguire questi passi:
1. **Installazione delle Dipendenze**: Prima di eseguire il programma, è necessario installare le dipendenze. Questo può essere fatto eseguendo il comando `pip install -r requirements.txt` nella directory principale del progetto. Questo comando installerà tutte le librerie necessarie, come numpy, pandas, matplotlib, etc. 
2. **Caricamento del Dataset**: 
Il programma è progettato per analizzare dataset provenienti da diversi formati di file. Per garantire il corretto funzionamento, il dataset deve soddisfare i seguenti requisiti:
### 1. Formato supportato
- Il file deve essere in uno dei seguenti formati:
  - `.csv`
  - `.xlsx` (Excel)
  - `.json`
  - `.txt` (con delimitatore `,`)
  - `.tsv` (con delimitatore `\t`)
- Se il formato del file non è tra quelli supportati, verrà generato un errore.
### 2. Struttura del dataset
- Deve contenere una colonna denominata **`Sample code number`**. Questa colonna sarà utilizzata per:
  - Identificare ogni campione univocamente.
  - Rimuovere eventuali duplicati basati su questa colonna.
  - Impostare questa colonna come indice del DataFrame.
### 3. Caricamento del dataset
- Durante l'esecuzione del programma, l'utente deve fornire il percorso del file contenente il dataset.
- Se non viene fornito alcun percorso o il file specificato non è valido, verrà utilizzato il dataset predefinito: `data/version_1.csv`.
### 4. Pulizia del dataset
- Dopo il caricamento, i duplicati nella colonna **`Sample code number`** verranno automaticamente rimossi.
- Il dataset risultante sarà indicizzato in base alla colonna **`Sample code number`**.
### **3. Configurazione Interattiva**
Durante l'esecuzione, il programma permette di configurare diverse fasi del processo attraverso opzioni interattive:
#### **Gestione dei Valori Mancanti**
L'utente può scegliere come trattare i valori mancanti nel dataset, selezionando una delle seguenti opzioni:
- `remove`: Elimina le righe con valori mancanti.
- `mean`: Sostituisce i valori mancanti con la media delle colonne.
- `median`: Sostituisce i valori mancanti con la mediana delle colonne (default).
- `mode`: Sostituisce i valori mancanti con il valore più frequente.
Se non viene fornita una scelta valida, il programma utilizza automaticamente la strategia `median`.
#### **Scaling delle Feature**
Per adattare i dati numerici, il programma offre due strategie di scaling:
- `normalize`: Normalizza i dati tra 0 e 1.
- `standardize`: Standardizza i dati con media 0 e deviazione standard 1.
Se non viene fornita una scelta valida, il programma utilizza `normalize` come default.
#### **Validazione del Modello**
L'utente può scegliere una strategia di validazione per dividere i dati in set di training e test:
1. **Holdout**: Divide i dati in due parti (training e test). È possibile specificare la percentuale di test (default: 20%).
2. **Random Subsampling**: Esegue più divisioni casuali del dataset. L'utente può specificare il numero di iterazioni (default: 10) e la percentuale di test (default: 20%).
3. **Leave-p-Out Cross Validation**: Esclude `p` campioni dal set di training in ogni iterazione. Il valore di `p` può essere specificato (default: 2).
### **4. Classificazione**
Il programma utilizza il classificatore **k-Nearest Neighbors (k-NN)** per distinguere tumori benigni e maligni.### **Scelta di \( k \) per il Classificatore k-NN**
Durante l'esecuzione del programma, l'utente può configurare il parametro \( k \), che rappresenta il numero di vicini da considerare per il classificatore k-Nearest Neighbors (k-NN). 
#### **Dettagli sulla Configurazione**
- Il valore di \( k \) determina quante osservazioni vicine verranno utilizzate per fare la classificazione.
- Se l'utente non specifica un valore, verrà utilizzato il valore di default: \( k = 3 \).
#### **Input dell'Utente**
Il programma richiederà di inserire il valore di \( k \) con il seguente messaggio:
 La fase di classificazione include:
1. **Preparazione dei Dati**
   - Le feature e le etichette vengono separate.
2. **Addestramento e Validazione**
   - Il classificatore k-NN viene addestrato sui dati di training e testato sui dati di test, in base alla strategia di validazione scelta:
     - **Holdout**
     - **Random Subsampling**
     - **Leave-p-Out Cross Validation**
3. **Predizione**
   - Il modello predice le etichette per i dati di test.
4. **Calcolo delle Metriche**
### **Metriche Calcolate**
Il progetto utilizza diverse metriche per valutare le prestazioni del modello di classificazione dei tumori. Le metriche disponibili sono:
- **`Accuracy Rate`**: La percentuale di predizioni corrette rispetto al totale. Valore ideale: vicino a 1.
- **`Error Rate`**: La percentuale di predizioni errate rispetto al totale. Valore ideale: vicino a 0.
- **`Sensitivity`** (o Recall): La capacità del modello di identificare correttamente i casi positivi (tumori maligni). Valore ideale: vicino a 1.
  - Formula: `Sensitivity = TP / (TP + FN)` dove:
    - `TP`: Veri positivi
    - `FN`: Falsi negativi
- **`Specificity`**: La capacità del modello di identificare correttamente i casi negativi (tumori benigni). Valore ideale: vicino a 1.
  - Formula: `Specificity = TN / (TN + FP)` dove:
    - `TN`: Veri negativi
    - `FP`: Falsi positivi
- **`Geometric Mean`**: Una misura dell'equilibrio tra Sensitivity e Specificity. Indica quanto il modello è bilanciato nell'identificazione delle due classi.
  - Formula: `Geometric Mean = √(Sensitivity × Specificity)`
- **`All the above`**: Opzione per calcolare e visualizzare tutte le metriche sopra elencate in una sola analisi.
Queste metriche forniscono una valutazione completa delle prestazioni del modello, sia in termini di accuratezza globale che di capacità di differenziare correttamente le due classi (positivi e negativi).
### **5. Visualizzazione e Salvataggio dei Risultati**
I risultati delle predizioni del modello verranno automaticamente salvati in un file Excel denominato `metrics_output.xlsx`. Questo file include metriche di performance come Accuracy, Sensitivity, Specificity e Geometric Mean, utili per analisi successive. Inoltre, verrà generato un grafico a barre per rappresentare visivamente le metriche.
### **Esempio di Output**
#### **1. Grafico delle Metriche**
Dopo l'esecuzione, verrà generato un grafico a barre che rappresenta le metriche calcolate. Di seguito un esempio di come potrebbe apparire.
Le metriche verranno salvate nel file `metrics_output.xlsx` con un formato simile a questo:
| Metric           | Value |
|-------------------|-------|
| Accuracy          | 0.85  |
| Sensitivity       | 0.88  |
| Specificity       | 0.83  |
| Geometric Mean    | 0.85  |
## **Conclusione**
Questo progetto rappresenta una pipeline completa e interattiva per la classificazione dei tumori, con un focus particolare sulla flessibilità e sulla facilità d'uso. Grazie allo script principale, è possibile:
- Caricare un dataset e gestire eventuali valori mancanti con strategie personalizzabili.
- Applicare tecniche di scaling delle feature per migliorare le prestazioni del modello.
- Selezionare diverse strategie di validazione per garantire una valutazione accurata.
- Utilizzare il classificatore k-Nearest Neighbors (k-NN) per distinguere tra tumori benigni e maligni.
- Visualizzare le metriche di performance attraverso grafici chiari e salvare i risultati in formato Excel per ulteriori analisi.
Questo strumento si rivela particolarmente utile per chi desidera esplorare e comprendere meglio le prestazioni dei modelli di machine learning applicati alla classificazione binaria. La modularità del codice consente di adattare facilmente le funzionalità a nuovi dataset o scenari specifici.
Ti invitiamo a esplorare lo script, personalizzare i parametri e utilizzare i risultati per approfondire le tue analisi. Non esitare a contattarci o contribuire con miglioramenti e nuove idee!
## **Contributi**
Per segnalare problemi, proporre miglioramenti o contribuire al progetto, puoi aprire un'issue o una pull request su GitHub. Ogni contributo è benvenuto e aiuterà a rendere il progetto ancora più utile