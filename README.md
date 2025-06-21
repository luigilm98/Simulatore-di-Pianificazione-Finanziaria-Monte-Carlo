# 📈 Simulatore di Pianificazione Finanziaria Monte Carlo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://simulatore-di-pianificazione-finanziaria-monte-carlo-amaeuqvh8.streamlit.app/)


## 🚀 Prova l'Applicazione Online

**[👉 CLICCA QUI PER PROVARE IL SIMULATORE ONLINE](https://simulatore-di-pianificazione-finanziaria-monte-carlo-amaeuqvh8.streamlit.app/)**

*Non serve installare nulla! L'applicazione funziona direttamente nel browser.*

---

## Come è nato questo progetto?

Non avendo competenze specifiche di programmazione, ho costruito questo simulatore da zero con l'aiuto di un'assistente AI avanzato (Gemini 2.5 Pro, tramite l'editor Cursor). Ho messo le idee e la logica, e l'AI mi ha aiutato a tradurle in codice Python funzionante. È stata un'esperienza incredibile, che dimostra come oggi la tecnologia possa aiutarci a realizzare progetti che un tempo sarebbero stati impensabili per chi non è del settore.

Non ho la presunzione di dire che sia perfetto, anzi, è un progetto in continua evoluzione. Tuttavia, l'ho costruito con impegno e ho grande fiducia che possa essere uno strumento utile per chiunque voglia prendere maggiore consapevolezza del proprio percorso finanziario.

Non sono un programmatore né un esperto di finanza, ho solo un diploma professionale e lavoro nel mio laboratorio odontotecnico. Da tempo seguo con enorme interesse il mondo della pianificazione finanziaria e dell'indipendenza economica, questo piccolo software è il risultato di un mix di idee, concetti e strategie che ho assorbito seguendo due figure che stimo tantissimo: **Mr. RIP** e **Paolo Coletti**. Ho cercato di prendere spunto dai loro insegnamenti per creare uno strumento pratico, un simulatore Monte Carlo che potesse aiutarmi (e spero anche voi) a visualizzare concretamente il futuro finanziario.

---

## 🎯 A Chi Serve Questo Strumento?

Questo simulatore è perfetto se ti sei mai chiesto:

*   "Quanto patrimonio avrò realisticamente quando smetterò di lavorare?"
*   "Qual è l'importo massimo che posso prelevare ogni anno in pensione senza finire i soldi troppo presto?"
*   "Voglio lasciare un'eredità o posso spendere tutto? E se sì, quanto?"
*   "L'impatto di tasse, bolli e costi sta frenando la mia crescita?"
*   "Il mio Fondo Pensione sta performando come dovrebbe? Che rendita posso aspettarmi?"

---

## 🔬 Come Funziona il Simulatore

### Il Motore Monte Carlo

Il cuore del simulatore è una **simulazione Monte Carlo** che esegue centinaia di "partite" del tuo futuro finanziario. Invece di darti una singola, ingannevole previsione, il software:

1. **Genera scenari multipli**: Per ogni simulazione, genera rendimenti e tassi di inflazione casuali basati su distribuzioni statistiche realistiche
2. **Simula l'intero percorso**: Calcola anno per anno l'evoluzione del tuo patrimonio, considerando tutti i fattori (investimenti, prelievi, tasse, costi)
3. **Analizza i risultati**: Dopo centinaia di simulazioni, ti mostra un ventaglio di possibilità, dal più sfortunato al più ottimistico

**Vantaggi rispetto alle previsioni lineari:**
- ✅ Mostra il **rischio reale** delle tue scelte
- ✅ Ti prepara agli **scenari peggiori**
- ✅ Ti permette di **ottimizzare** la strategia
- ✅ È **scientificamente fondato** su dati storici

### Modello Finanziario Italiano

Il simulatore è specificamente progettato per il contesto italiano:

#### 📊 Gestione Investimenti
- **Portafoglio ETF Personalizzabile**: Costruisci il tuo portafoglio con ETF reali, specificando allocazioni e TER
- **Ribilanciamento Automatico**: Il software mantiene le tue allocazioni target vendendo/acquistando automaticamente
- **Calcolo Rendimenti Realistici**: Usa distribuzioni log-normali basate su dati storici del mercato italiano

#### 💰 Strategie di Prelievo Intelligenti
1. **FISSO**: Prelevi un importo fisso annuale, corretto per l'inflazione
2. **REGOLA_4_PERCENTO**: Prelevi una percentuale fissa del patrimonio all'inizio di ogni anno
3. **GUARDRAIL**: Versione intelligente che adatta i prelievi in base alla performance del mercato

#### 🏛️ Tassazione Italiana Integrata
- **Capital Gain (26%)**: Calcolata su ogni plusvalenza da vendita ETF
- **Imposta di Bollo**: 
  - 0.20% annuo sui titoli
  - 34.20€ su conti correnti >5.000€
- **Tassazione Fondo Pensione**: 
  - 20% sui rendimenti annuali
  - Aliquota finale sul capitale ritirato

#### 🏦 Modulo Fondo Pensione Completo
- **Accumulo con TER**: Simula costi reali del fondo
- **Liquidazione Separata**: Gestisce capitale ritirato vs. rendita
- **Tassazione Differenziata**: Applica le aliquote corrette per ogni componente

#### 📈 Asset Allocation Dinamica (Glidepath)
- **Riduzione Progressiva del Rischio**: Sposta gradualmente da ETF a liquidità con l'avanzare dell'età
- **Protezione del Capitale**: Mantiene una riserva di liquidità per emergenze
- **Ottimizzazione Automatica**: Calcola le allocazioni ottimali anno per anno

### Interfaccia Web Interattiva

#### 🎛️ Sidebar di Controllo
- **Parametri Demografici**: Età, orizzonte temporale, età di pensionamento
- **Situazione Finanziaria**: Patrimonio iniziale, risparmi mensili, pensioni attese
- **Configurazione Investimenti**: Rendimenti, volatilità, inflazione, costi
- **Strategia di Prelievo**: Tipo di strategia e parametri specifici
- **Costruttore Portafoglio**: Tabella interattiva per definire ETF e allocazioni

#### 📊 Dashboard dei Risultati
- **Statistiche Principali**: Patrimonio finale mediano, probabilità di successo, prelievo sostenibile
- **Grafici Interattivi**: 
  - Spaghetti plot delle simulazioni
  - Istogramma del patrimonio finale
  - Evoluzione del patrimonio nel tempo
- **Analisi per Fasi**: Separazione tra fase di accumulo e decumulo

#### 🔍 Analisi Dettagliata
- **Tab "Fase di Accumulo"**: Focus sui risparmi e crescita del patrimonio
- **Tab "Fase di Decumulo"**: Analisi dei prelievi e sostenibilità
- **Spiegazioni "For Dummies"**: Traduzione in linguaggio semplice dei risultati tecnici

---

## 🚀 Installazione e Utilizzo

### 🎯 Opzione 1: Usa Online (Consigliato)

**Non serve installare nulla!** L'applicazione è disponibile online:

**[👉 PROVA SUBITO ONLINE](https://simulatore-di-pianificazione-finanziaria-monte-carlo-amaeuqvh8.streamlit.app/)**

### 💻 Opzione 2: Installa Localmente

Se preferisci eseguire l'applicazione sul tuo computer:

#### Prerequisiti
- [Python 3.8+](https://www.python.org/downloads/)
- Connessione internet (per installare le dipendenze)

#### Installazione Rapida

```bash
# 1. Clona il repository
git clone https://github.com/luigilm98/Simulatore-di-Pianificazione-Finanziaria-Monte-Carlo.git
cd Simulatore-di-Pianificazione-Finanziaria-Monte-Carlo

# 2. Crea ambiente virtuale
python3 -m venv venv
source venv/bin/activate  # Su Windows: venv\Scripts\activate

# 3. Installa dipendenze
pip install -r requirements.txt

# 4. Avvia l'applicazione
streamlit run app.py
```

L'applicazione si aprirà automaticamente nel browser all'indirizzo `http://localhost:8501`

### Guida all'Uso

1. **Configura i Parametri Base**: Inizia dalla sidebar sinistra, inserendo la tua situazione attuale
2. **Costruisci il Portafoglio**: Usa la tabella ETF per definire le tue allocazioni
3. **Scegli la Strategia**: Seleziona la strategia di prelievo più adatta a te
4. **Analizza i Risultati**: Esplora i grafici e le statistiche per capire le implicazioni
5. **Sperimenta**: Modifica i parametri in tempo reale per vedere come cambiano i risultati

---

## 🛠️ Tecnologie Utilizzate

*   **Linguaggio:** Python 3.8+
*   **Interfaccia Web:** [Streamlit](https://streamlit.io/) - Framework per app web interattive
*   **Calcolo Numerico:** [NumPy](https://numpy.org/) & [Pandas](https://pandas.pydata.org/) - Analisi dati e calcoli scientifici
*   **Grafici Interattivi:** [Plotly](https://plotly.com/) - Visualizzazioni dinamiche e responsive
*   **Sviluppo Assistito da AI:** [Cursor](https://cursor.sh/) (con Gemini 2.5 Pro) - Editor intelligente

---

## 📚 Concetti Chiave Spiegati

### Monte Carlo Simulation
La simulazione Monte Carlo è una tecnica matematica che usa la casualità per risolvere problemi deterministici. Nel nostro caso, genera centinaia di scenari futuri possibili per il mercato finanziario, permettendoci di vedere non solo il risultato "più probabile", ma anche tutti i possibili esiti e le loro probabilità.

### Regola del 4%
La "Regola del 4%" suggerisce che puoi prelevare in sicurezza il 4% del tuo patrimonio iniziale ogni anno, aumentandolo per l'inflazione. Il nostro simulatore testa questa regola e le sue varianti nel contesto italiano.

### Glidepath
Il "glidepath" (percorso di discesa) è una strategia che riduce gradualmente l'esposizione al rischio con l'avanzare dell'età, spostando il patrimonio da investimenti azionari a obbligazionari/liquidità.

### TER (Total Expense Ratio)
Il TER rappresenta i costi annuali di gestione di un fondo o ETF, espressi come percentuale del patrimonio investito. Costi più bassi significano rendimenti netti più alti.

---

## ⚠️ Disclaimer e Limitazioni

**Questo è uno strumento creato a scopo educativo. Non è una consulenza finanziaria.**

### Limitazioni del Modello
- **Dati Storici**: Le simulazioni si basano su dati storici che potrebbero non ripetersi
- **Semplificazioni**: Il modello non include tutti i fattori della vita reale (es. spese impreviste, cambiamenti normativi)
- **Assunzioni**: I rendimenti futuri sono stimati, non garantiti

### Raccomandazioni
- Usa il simulatore per **esplorare scenari** e **farti domande migliori**
- Consulta sempre un **professionista qualificato** prima di prendere decisioni finanziarie
- Considera il simulatore come un **punto di partenza**, non come una risposta definitiva

---

## 🤝 Contributi e Deploy Personale

Questo progetto è open source e i contributi sono benvenuti!

### Contribuire al Progetto
Se hai idee per miglioramenti, correzioni o nuove funzionalità, non esitare a:

1. Aprire una [Issue](https://github.com/luigilm98/Simulatore-di-Pianificazione-Finanziaria-Monte-Carlo/issues) per segnalare bug o suggerire miglioramenti
2. Fare un [Pull Request](https://github.com/luigilm98/Simulatore-di-Pianificazione-Finanziaria-Monte-Carlo/pulls) con le tue modifiche

### Deployare la Tua Versione
Se vuoi sperimentare con il codice o deployare la tua versione personale dell'app, puoi farlo con un solo click:

[![Deploy to Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=luigilm98/Simulatore-di-Pianificazione-Finanziaria-Monte-Carlo&branch=main&mainModule=app.py)

---

## 📄 Licenza

Questo progetto è rilasciato sotto licenza [MIT](LICENSE). Puoi usare, modificare e distribuire liberamente questo software.

---

*Grazie a Mr. RIP e Paolo Coletti per l'ispirazione e gli insegnamenti che hanno reso possibile questo progetto.* 
