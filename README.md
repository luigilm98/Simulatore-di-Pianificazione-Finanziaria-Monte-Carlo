# 📈 Simulatore di Pianificazione Finanziaria Monte Carlo v3.0

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://simulatore-di-pianificazione-finanziaria-monte-carlo-amaeuqvh8.streamlit.app/)

Questo strumento è progettato per aiutarti a navigare la complessità del tuo futuro finanziario attraverso simulazioni potenti e visualizzazioni intuitive.

---

## Indice
1. [Filosofia del Progetto](#-filosofia-del-progetto)
2. [Guida Rapida: Prova Online](#-guida-rapida-prova-online)
3. [La Logica del Simulatore: Un'Analisi Dettagliata](#-la-logica-del-simulatore-unanalisi-dettagliata)
    - [Il Motore Monte Carlo](#il-motore-monte-carlo)
    - [Il Ciclo di Vita Mensile: Cosa Succede in una Simulazione](#il-ciclo-di-vita-mensile-cosa-succede-in-una-simulazione)
    - [Dizionario dei Parametri di Input](#-dizionario-dei-parametri-di-input)
    - [Le Formule Chiave Spiegate](#-le-formule-chiave-spiegate)
4. [Installazione Locale](#-installazione-locale)
5. [Tecnologie Utilizzate](#-tecnologie-utilizzate)
6. [Disclaimer e Limitazioni](#-disclaimer-e-limitazioni)
7. [Licenza](#-licenza)

---

## 💡 Filosofia del Progetto

*Questa sezione è stata scritta dall'autore originale del progetto.*

Non avendo competenze specifiche di programmazione, ho costruito questo simulatore da zero con l'aiuto di un'assistente AI avanzato (Gemini 2.5 Pro, tramite l'editor Cursor). Ho messo le idee e la logica, e l'AI mi ha aiutato a tradurle in codice Python funzionante.

Non sono un programmatore né un esperto di finanza, ho solo un diploma professionale e lavoro nel mio laboratorio odontotecnico. Da tempo seguo con enorme interesse il mondo della pianificazione finanziaria e dell'indipendenza economica. Questo piccolo software è il risultato di un mix di idee, concetti e strategie che ho assorbito seguendo due figure che stimo tantissimo: **Mr. RIP** e **Paolo Coletti**. Ho cercato di prendere spunto dai loro insegnamenti per creare uno strumento pratico, un simulatore Monte Carlo che potesse aiutarmi (e spero anche voi) a visualizzare concretamente il futuro finanziario.

---

## 🚀 Guida Rapida: Prova Online

Il modo più semplice per usare il simulatore è tramite la versione web, senza installare nulla.

**[👉 CLICCA QUI PER PROVARE IL SIMULATORE ONLINE](https://simulatore-di-pianificazione-finanziaria-monte-carlo-amaeuqvh8.streamlit.app/)**

---

## 🔬 La Logica del Simulatore: Un'Analisi Dettagliata

### Il Motore Monte Carlo
Il cuore del simulatore non è una singola previsione, ma un motore statistico che esegue migliaia di "vite finanziarie" parallele. Per ogni vita, genera rendimenti degli investimenti e tassi di inflazione leggermente diversi, basati su modelli statistici standard (distribuzione log-normale per i rendimenti, distribuzione normale per l'inflazione). Questo approccio permette di visualizzare non solo lo scenario "medio", ma l'intero ventaglio di possibilità, dal più fortunato al più sfortunato, fornendo una visione onesta e realistica del rischio.

### Il Ciclo di Vita Mensile: Cosa Succede in una Simulazione
Ogni simulazione avanza mese per mese. Ecco l'ordine esatto delle operazioni che il motore esegue per ogni mese simulato:

1.  **Contribuisci e Investi (Fase di Accumulo):** Se sei prima della pensione, il motore aggiunge i tuoi contributi mensili al conto corrente e al portafoglio ETF.
2.  **Gestisci il Fondo Pensione:** Se attivo, il contributo mensile viene versato nel fondo pensione.
3.  **Incassa le Rendite Passive:** Se hai raggiunto l'età per la pensione pubblica o per la rendita del fondo pensione, gli importi mensili vengono accreditati sul conto corrente, adeguati all'inflazione.
4.  **Calcola il Fabbisogno per i Prelievi (Fase di Decumulo):** Se sei in pensione, il motore calcola l'importo mensile da prelevare in base alla strategia scelta (Regola del 4%, Fisso, Guardrail).
5.  **Vendi per Prelevare (se necessario):** Se la liquidità sul conto non basta a coprire il prelievo, il motore vende la quantità minima necessaria di ETF per generare la liquidità richiesta, pagando la tassazione sul capital gain (26%) sulla plusvalenza realizzata.
6.  **Esegui il Prelievo:** L'importo calcolato viene sottratto dal conto corrente.
7.  **Applica i Rendimenti di Mercato:** Il valore del portafoglio ETF e del fondo pensione viene aggiornato applicando un rendimento mensile casuale (ma statisticamente plausibile).
8.  **Aggiorna l'Inflazione:** L'indice generale dei prezzi viene aggiornato con un tasso di inflazione mensile casuale.
9.  **Applica Costi e Tasse di Fine Anno (solo a Dicembre):**
    *   **Costi ETF:** Vengono detratti i costi fissi mensili e, a fine anno, il TER (Total Expense Ratio).
    *   **Tasse FP:** Viene applicata l'imposta del 20% sui rendimenti maturati nel fondo pensione.
    *   **Imposta di Bollo:** Viene applicata l'imposta di bollo (0.2% sui titoli, 34.20€ sul conto corrente se sopra i 5k€).
    *   **Ribilanciamento (Glidepath):** Se il Glidepath è attivo e l'allocazione attuale non rispetta quella target per la tua età, il motore vende o compra ETF per tornare all'allocazione desiderata.

Questo ciclo si ripete per tutta la durata della simulazione, per ognuna delle migliaia di "vite" parallele.

### 📘 Dizionario dei Parametri di Input

Ogni parametro nella sidebar ha un impatto preciso sulla simulazione.

#### 1. Parametri di Base
- **Età Iniziale, Capitale, ETF, Contributi:** Definiscono il tuo punto di partenza e la tua capacità di risparmio.
- **Inflazione Media Annua:** È il "nemico silenzioso". Un'inflazione più alta ridurrà il potere d'acquisto futuro del tuo patrimonio.
- **Anni all'Inizio dei Prelievi:** Determina la durata della tua fase di accumulo. Più è lunga, più il capitale avrà tempo di crescere.
- **Numero Simulazioni / Orizzonte Temporale:** Parametri tecnici. Più simulazioni esegui, più il risultato sarà statisticamente affidabile.

#### 2. Costruttore di Portafoglio ETF
- **Allocazione (%):** Definisce il peso di ogni strumento nel tuo portafoglio.
- **TER (%):** Il costo annuo di ogni ETF. Viene sottratto direttamente dai rendimenti.
- **Rendimento/Volatilità Attesa (%):** Sono le tue stime sul comportamento futuro dei mercati. Il simulatore usa questi valori come media per generare i rendimenti casuali. La volatilità determina l'ampiezza delle oscillazioni (più alta è, più il "cono di incertezza" sarà largo).

#### 3. Strategie di Prelievo
- **FISSO:** Prelevi ogni anno lo stesso importo, adeguato all'inflazione. Strategia rigida, può fallire se il mercato va male.
- **REGOLA 4%:** Prelevi ogni anno il 4% del capitale *rimanente*. È più flessibile: prelevi meno quando il mercato scende e di più quando sale.
- **GUARDRAIL:** Una versione intelligente della Regola del 4%. Adeguata i prelievi all'inflazione, ma con dei "paraurti" (guardrail): se il prelievo diventa una frazione troppo alta o troppo bassa del capitale, viene corretto (es. tagliato o aumentato del 10%) per evitare di esaurire il capitale o di vivere troppo frugalmente.

#### 4. Asset Allocation Dinamica (Glidepath)
- **Logica:** Serve a ridurre il rischio con l'avanzare dell'età. Sposta gradualmente il patrimonio da investimenti più rischiosi (ETF) a liquidità (Conto Corrente) tra l'anno di inizio e fine del glidepath.
- **Impatto:** Riduce la volatilità del portafoglio in pensione, proteggendoti dal *Sequence of Returns Risk*.

#### 5. Tassazione e Costi
- **Tassazione Capital Gain:** Viene applicata solo quando vendi ETF in profitto.
- **Imposta di Bollo:** Una tassa fissa sul patrimonio, che agisce come un costo aggiuntivo.
- **Costo Fisso Deposito Titoli:** Un costo amministrativo che erode leggermente il capitale.

#### 6. Fondo Pensione
- **Logica:** Simula un piano pensionistico complementare. I contributi crescono con un proprio rendimento e volatilità.
- **Tassazione Agevolata:** I rendimenti vengono tassati al 20% (invece del 26% standard). Al momento del ritiro, il capitale accumulato viene tassato con un'aliquota finale agevolata (tipicamente tra il 9% e il 15%).
- **Ritiro Capitale vs Rendita:** Puoi scegliere di ritirare una parte del montante subito (pagando l'aliquota finale) e trasformare il resto in una rendita annua per il futuro.

#### 7. Altre Entrate
- **Pensione Pubblica:** Una rendita esterna che si aggiunge alle tue entrate durante la pensione, sostenendo il tuo tenore di vita.

### 🧮 Le Formule Chiave Spiegate

- **Potere d'Acquisto (Valore Reale):** `Valore Reale = Valore Nominale / Indice dei Prezzi`. L'indice dei prezzi parte da 1 e cresce ogni mese con l'inflazione. Questa formula permette di capire quanto varranno i tuoi soldi in termini di "Euro di oggi".
- **Tassazione sulla Plusvalenza (ETF):** `Tasse = (Prezzo di Vendita - Prezzo di Acquisto) * 0.26`. Viene calcolata su un *cost basis* medio che tiene traccia di tutti i tuoi acquisti.
- **Rendimento Netto di un Asset:** `Rendimento Netto = Rendimento Lordo - TER - Tasse (se applicabili) - Imposta di Bollo`.
- **Conversione Montante FP in Rendita:** Utilizza una formula attuariale standard per calcolare la rata annua che puoi ottenere da un capitale, data una durata e un tasso di rendimento atteso.

---

## 💾 Installazione Locale

Se preferisci eseguire l'applicazione sul tuo computer:

#### Prerequisiti
- [Python 3.8+](https://www.python.org/downloads/)
- [Git](https://git-scm.com/downloads)

#### Installazione Rapida
```bash
# 1. Clona il repository
git clone https://github.com/luigilm98/Simulatore-di-Pianificazione-Finanziaria-Monte-Carlo.git
cd Simulatore-di-Pianificazione-Finanziaria-Monte-Carlo

# 2. Crea e attiva un ambiente virtuale
    python3 -m venv venv
source venv/bin/activate  # Su Windows: venv\Scripts\activate

# 3. Installa le dipendenze
    pip install -r requirements.txt

# 4. Avvia l'applicazione
streamlit run app.py
```
L'applicazione si aprirà automaticamente nel browser all'indirizzo `http://localhost:8501`.

---

## 🛠️ Tecnologie Utilizzate
*   **Linguaggio:** Python
*   **Interfaccia Web:** Streamlit
*   **Calcolo Numerico:** NumPy & Pandas
*   **Grafici Interattivi:** Plotly
*   **Sviluppo Assistito da AI:** Cursor (con Gemini 2.5 Pro)

---

## ⚠️ Disclaimer e Limitazioni
**Questo è uno strumento creato a scopo educativo e di intrattenimento. Non è una consulenza finanziaria.** Le simulazioni si basano su modelli statistici che non possono prevedere il futuro. Usa il simulatore per farti domande migliori e per capire l'impatto delle tue scelte, ma consulta sempre un professionista qualificato prima di prendere decisioni finanziarie reali.

---

## 📄 Licenza
Questo progetto è rilasciato sotto licenza [MIT](LICENSE).

---
*Grazie a Mr. RIP e Paolo Coletti per l'ispirazione e gli insegnamenti che hanno reso possibile questo progetto.* 