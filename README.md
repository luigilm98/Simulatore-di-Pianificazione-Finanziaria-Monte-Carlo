# 📈 Simulatore di Pianificazione Finanziaria con Modello Economico a Regimi

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://simulatore-di-pianificazione-finanziaria-monte-carlo-amaeuqvh8.streamlit.app/)

Questo strumento integra un **Modello Economico a Regimi Commutabili (Regime-Switching Model)** per generare scenari futuri realistici, simulando veri e propri cicli economici completi di crash, recessioni e riprese.

---

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

Questo simulatore è perfetto se vuoi andare oltre le proiezioni lineari e ti sei mai chiesto:

*   "Il mio piano di accumulo sopravviverebbe a una crisi come quella del 2008 seguita da anni di stagnazione?"
*   "Cosa succederebbe al mio patrimonio se affrontassimo un decennio di alta inflazione e bassi rendimenti come negli anni '70?"
*   "Qual è l'impatto reale di tasse, bolli e costi sul mio patrimonio finale in diversi scenari economici?"
*   "Qual è il tasso di prelievo *veramente* sicuro per la mia pensione, considerando la possibilità di crisi prolungate?"

---

## 🔬 Come Funziona il Simulatore: Il Vantaggio del Modello a Regimi

La maggior parte dei simulatori finanziari usa un modello "random walk", dove ogni anno è un evento casuale indipendente. Il mondo reale non funziona così. Le crisi non sono eventi isolati, ma fasi di un ciclo.

Questo simulatore supera questo limite implementando un **Modello Economico a Regimi Commutabili**.

### 1. Definizione dei Regimi Economici
Invece di un singolo set di parametri (rendimento medio, volatilità), il simulatore definisce diversi **stati economici** (o "regimi"), ciascuno con le proprie caratteristiche:

*   **Regimi di Mercato**:
    *   `Normal`: Crescita stabile, bassa volatilità.
    *   `Crash`: Crollo improvviso, altissima volatilità.
    *   `Recession`: Rendimenti negativi o piatti, alta volatilità.
    *   `Recovery`: Forte ripresa dopo una crisi, alta volatilità.
*   **Regimi di Inflazione**:
    *   `Normal`: Inflazione controllata intorno al 2-3%.
    *   `High`: Inflazione elevata e volatile.
    *   `Deflation`: Inflazione negativa.

### 2. Matrice di Transizione
Il cuore del modello è una **matrice di probabilità di transizione**. Questa matrice definisce la probabilità che, dato lo stato economico di quest'anno, si passi a un altro stato l'anno successivo.

*Esempio*:
*   Se siamo in un regime di `Crash`, c'è una probabilità del 100% di passare a un regime di `Recession` l'anno successivo.
*   Se siamo in `Recession`, c'è una probabilità del 95% di *rimanere* in `Recession` e una del 5% di passare a `Recovery`.

Questo crea **memoria e dipendenza temporale**, generando cicli economici molto più realistici rispetto a shock casuali e indipendenti.

### 3. Scenari Pre-configurati
L'utente non deve impostare manualmente questi parametri complessi. Può semplicemente scegliere da un menu a tendina tra diversi **modelli macroeconomici pre-configurati**:

*   **Volatile (Cicli Boom-Bust)**: Il nostro modello base con cicli di mercato pronunciati.
*   **Stabilità (Crescita Lenta)**: Un'economia con bassa volatilità e crescita modesta.
*   **Stagflazione Anni '70**: Simula un'economia con alta inflazione e rendimenti reali negativi.
*   **Crisi Prolungata (Giappone)**: Modella un lungo periodo di stagnazione e deflazione.

### Vantaggi di Questo Approccio
- ✅ **Realismo Superiore**: Genera sequenze di rendimenti che assomigliano a veri cicli economici storici.
- ✅ **Stress Test Efficaci**: Permette di testare la resilienza di un piano finanziario contro scenari avversi complessi e prolungati.
- ✅ **Comprensione Intuitiva**: L'utente può testare il proprio piano contro scenari noti ("Anni '70", "Crisi Giapponese") senza dover manipolare decine di parametri.

---

## 🛠️ Altre Funzionalità Chiave

Oltre al motore economico, il simulatore include un'analisi finanziaria completa e specifica per l'**Italia**:

*   **Costruttore di Portafoglio ETF**: Personalizza il tuo portafoglio e calcola automaticamente i parametri di rischio/rendimento e costi (TER).
*   **Modulo Fondo Pensione Dettagliato**: Simula l'accumulo, la tassazione agevolata sui rendimenti, la liquidazione parziale e la conversione in rendita.
*   **Strategie di Prelievo Avanzate**: Scegli tra importo `FISSO`, la classica `REGOLA DEL 4%` o la strategia adattiva `GUARDRAIL`.
*   **Tassazione Italiana Integrata**: Calcola automaticamente l'imposta di bollo (titoli e conto), la tassazione sul capital gain (26%) e le aliquote fiscali specifiche per il fondo pensione.
*   **Asset Allocation Dinamica (Glidepath)**: Imposta una riduzione automatica e progressiva del rischio con l'avvicinarsi della pensione.

---

## 🚀 Installazione e Utilizzo

### 🎯 Opzione 1: Usa Online (Consigliato)

**Non serve installare nulla!** L'applicazione è disponibile online:

**[👉 PROVA SUBITO ONLINE](https://simulatore-di-pianificazione-finanziaria-monte-carlo-amaeuqvh8.streamlit.app/)**

### 💻 Opzione 2: Installa Localmente

Se preferisci eseguire l'applicazione sul tuo computer:

#### Prerequisiti
- [Python 3.8+](https://www.python.org/downloads/)
- `git` per clonare il repository.

#### Installazione Rapida

```bash
# 1. Clona il repository
git clone https://github.com/luigilm98/Simulatore-di-Pianificazione-Finanziaria-Monte-Carlo.git
cd Simulatore-di-Pianificazione-Finanziaria-Monte-Carlo

# 2. Crea un ambiente virtuale (consigliato)
python3 -m venv venv
source venv/bin/activate  # Su Windows: venv\Scripts\activate

# 3. Installa le dipendenze
pip install -r requirements.txt

# 4. Avvia l'applicazione
streamlit run app.py
```
L'applicazione si aprirà automaticamente nel browser all'indirizzo `http://localhost:8501`.

### Guida all'Uso

1. **Configura i Parametri Base**: Inizia dalla sidebar sinistra, inserendo la tua situazione attuale
2. **Costruisci il Portafoglio**: Usa la tabella ETF per definire le tue allocazioni
3. **Scegli la Strategia**: Seleziona la strategia di prelievo più adatta a te
4. **Analizza i Risultati**: Esplora i grafici e le statistiche per capire le implicazioni
5. **Sperimenta**: Modifica i parametri in tempo reale per vedere come cambiano i risultati

---

## 🤝 Contributi

Questo progetto è open source e i contributi sono i benvenuti. Se hai idee per miglioramenti, correzioni o nuovi modelli economici, sentiti libero di aprire una [Issue](https://github.com/luigilm98/Simulatore-di-Pianificazione-Finanziaria-Monte-Carlo/issues) o un [Pull Request](https://github.com/luigilm98/Simulatore-di-Pianificazione-Finanziaria-Monte-Carlo/pulls).

---

## 📄 Licenza

Questo progetto è rilasciato sotto licenza [MIT](LICENSE).

---
*Un ringraziamento speciale a Mr. RIP e Paolo Coletti, le cui idee e insegnamenti sono stati la fonte di ispirazione principale per questo progetto.*
