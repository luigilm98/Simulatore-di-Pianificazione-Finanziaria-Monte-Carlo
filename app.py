import streamlit as st
import numpy as np
import simulation_engine as engine
import plotly.graph_objects as go
import pandas as pd
import json
import os
from datetime import datetime

# --- Gestione File e Dati ---
HISTORY_DIR = "simulation_history"
os.makedirs(HISTORY_DIR, exist_ok=True)

class NumpyEncoder(json.JSONEncoder):
    """ Encoder JSON speciale per tipi di dati NumPy. """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def save_simulation(name, params, results):
    """Salva i parametri e i risultati di una simulazione in un file JSON."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{name.replace(' ', '_')}.json"
    filepath = os.path.join(HISTORY_DIR, filename)
    
    data_to_save = {
        "name": name,
        "timestamp": timestamp,
        "parameters": params,
        "results": results
    }
    
    with open(filepath, 'w') as f:
        json.dump(data_to_save, f, cls=NumpyEncoder, indent=4)
    st.success(f"Simulazione '{name}' salvata con successo!")

def load_simulations():
    """Carica i metadati di tutte le simulazioni salvate."""
    simulations = []
    for filename in sorted(os.listdir(HISTORY_DIR), reverse=True):
        if filename.endswith(".json"):
            try:
                filepath = os.path.join(HISTORY_DIR, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    simulations.append({
                        "name": data.get("name", "Senza nome"),
                        "timestamp": data.get("timestamp", "N/A"),
                        "filename": filename
                    })
            except (json.JSONDecodeError, IOError) as e:
                st.warning(f"Impossibile caricare {filename}: {e}")
    return simulations

def load_simulation_data(filename):
    """Carica i dati completi di una specifica simulazione."""
    filepath = os.path.join(HISTORY_DIR, filename)
    with open(filepath, 'r') as f:
        return json.load(f)

def delete_simulation(filename):
    """Elimina un file di simulazione salvato."""
    filepath = os.path.join(HISTORY_DIR, filename)
    os.remove(filepath)
    st.rerun()

# --- Funzioni di Plotting ---
def hex_to_rgb(hex_color):
    """Converte un colore esadecimale in una tupla RGB."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def plot_percentile_chart(data, title, y_title, color_median, color_fill, anni_totali):
    """Crea un grafico a 'cono' con i percentili."""
    fig = go.Figure()
    anni_asse_x = np.linspace(0, anni_totali, data.shape[1])
    
    # Calcolo percentili
    p10 = np.percentile(data, 10, axis=0)
    p25 = np.percentile(data, 25, axis=0)
    median_data = np.median(data, axis=0)
    p75 = np.percentile(data, 75, axis=0)
    p90 = np.percentile(data, 90, axis=0)

    rgb_fill = hex_to_rgb(color_fill)

    # Area 10-90 percentile (range più ampio)
    fig.add_trace(go.Scatter(
        x=np.concatenate([anni_asse_x, anni_asse_x[::-1]]),
        y=np.concatenate([p90, p10[::-1]]),
        fill='toself',
        fillcolor=f'rgba({rgb_fill[0]}, {rgb_fill[1]}, {rgb_fill[2]}, 0.2)',
        line={'color': 'rgba(255,255,255,0)'},
        name='10-90 Percentile',
        hoverinfo='none'
    ))

    # Area 25-75 percentile (range interquartile)
    fig.add_trace(go.Scatter(
        x=np.concatenate([anni_asse_x, anni_asse_x[::-1]]),
        y=np.concatenate([p75, p25[::-1]]),
        fill='toself',
        fillcolor=f'rgba({rgb_fill[0]}, {rgb_fill[1]}, {rgb_fill[2]}, 0.4)',
        line={'color': 'rgba(255,255,255,0)'},
        name='25-75 Percentile',
        hoverinfo='none'
    ))

    # Linea mediana
    fig.add_trace(go.Scatter(
        x=anni_asse_x, y=median_data, mode='lines',
        name='Scenario Mediano (50°)',
        line={'width': 3, 'color': color_median},
        hovertemplate='Anno %{x:.1f}<br>Patrimonio Mediano: €%{y:,.0f}<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Anni",
        yaxis_title=y_title,
        yaxis_tickformat="€,d",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig

def plot_histogram(data, anni_totali):
    fig = go.Figure(data=[go.Histogram(x=data, nbinsx=30, marker_color='#4472C4')])
    fig.update_layout(
        title_text='Distribuzione del Patrimonio Finale Reale',
        xaxis_title_text='Patrimonio Finale (Potere d\'Acquisto Odierno)',
        yaxis_title_text='Numero di Simulazioni',
        bargap=0.1
    )
    return fig

def plot_success_probability(data, anni_totali):
    anni_grafico = np.arange(data.size)
    fig = go.Figure(data=go.Scatter(
        x=anni_grafico, y=data, mode='lines+markers', 
        line=dict(color='#C00000', width=3),
        hovertemplate='Anno %{x}:<br>Successo: %{y:.0%}<extra></extra>'
    ))
    fig.update_layout(
        title='Probabilità di Successo nel Tempo',
        xaxis_title='Anni di Simulazione',
        yaxis_title='Probabilità di Avere Patrimonio Residuo',
        yaxis_tickformat='.0%',
        yaxis_range=[0, 1.01]
    )
    return fig

def plot_income_composition(data, anni_totali):
    anni_asse_x_annuale = np.arange(anni_totali)
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=anni_asse_x_annuale, y=data['prelievi_effettivi_reali'],
        name='Prelievi dal Patrimonio', stackgroup='one',
        line={'color': '#4472C4'}
    ))
    fig.add_trace(go.Scatter(
        x=anni_asse_x_annuale, y=data['pensioni_pubbliche_reali'],
        name='Pensione Pubblica', stackgroup='one',
        line={'color': '#ED7D31'}
    ))
    fig.add_trace(go.Scatter(
        x=anni_asse_x_annuale, y=data['rendite_fp_reali'],
        name='Rendita Fondo Pensione', stackgroup='one',
        line={'color': '#A5A5A5'}
    ))

    fig.update_layout(
        title='Composizione del Reddito Annuo Reale (Scenario Mediano)',
        xaxis_title='Anni',
        yaxis_title='Reddito Reale Annuo (€ Odierni)',
        yaxis_tickformat="€,d",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig

def plot_asset_allocation(data, anni_totali):
    anni_asse_x_annuale = np.arange(anni_totali)
    banca_reale = data['saldo_banca_reale']
    etf_reale = data['saldo_etf_reale']
    fp_reale = data['saldo_fp_reale']
    
    totale_reale = banca_reale + etf_reale + fp_reale
    with np.errstate(divide='ignore', invalid='ignore'):
        banca_perc = np.nan_to_num(banca_reale / totale_reale)
        etf_perc = np.nan_to_num(etf_reale / totale_reale)
        fp_perc = np.nan_to_num(fp_reale / totale_reale)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=anni_asse_x_annuale, y=banca_perc,
        name='Liquidità', stackgroup='one', groupnorm='percent',
        line={'color': '#5B9BD5'}
    ))
    fig.add_trace(go.Scatter(
        x=anni_asse_x_annuale, y=etf_perc,
        name='ETF', stackgroup='one',
        line={'color': '#ED7D31'}
    ))
    fig.add_trace(go.Scatter(
        x=anni_asse_x_annuale, y=fp_perc,
        name='Fondo Pensione', stackgroup='one',
        line={'color': '#70AD47'}
    ))

    fig.update_layout(
        title='Allocazione % del Patrimonio (Scenario Mediano)',
        xaxis_title='Anni',
        yaxis_title='Percentuale del Patrimonio Totale',
        yaxis_tickformat='.0%',
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig

def plot_income_cone_chart(data, anni_totali, anni_inizio_prelievo):
    """Crea un grafico a 'cono' per il reddito reale annuo."""
    fig = go.Figure()
    # Mostra i dati solo a partire dall'anno di inizio prelievo
    start_index = int(anni_inizio_prelievo)
    if start_index >= data.shape[1]:
        return fig # Non c'è nulla da plottare se l'inizio è oltre l'orizzonte

    anni_asse_x = np.arange(start_index, anni_totali + 1)
    data_decumulo = data[:, start_index:]

    p10 = np.percentile(data_decumulo, 10, axis=0)
    p25 = np.percentile(data_decumulo, 25, axis=0)
    median_data = np.median(data_decumulo, axis=0)
    p75 = np.percentile(data_decumulo, 75, axis=0)
    p90 = np.percentile(data_decumulo, 90, axis=0)

    # Area 10-90
    fig.add_trace(go.Scatter(
        x=np.concatenate([anni_asse_x, anni_asse_x[::-1]]),
        y=np.concatenate([p90, p10[::-1]]),
        fill='toself',
        fillcolor='rgba(0, 176, 246, 0.2)',
        line={'color': 'rgba(255,255,255,0)'},
        name='10-90 Percentile',
        hoverinfo='none'
    ))
    # Area 25-75
    fig.add_trace(go.Scatter(
        x=np.concatenate([anni_asse_x, anni_asse_x[::-1]]),
        y=np.concatenate([p75, p25[::-1]]),
        fill='toself',
        fillcolor='rgba(0, 176, 246, 0.4)',
        line={'color': 'rgba(255,255,255,0)'},
        name='25-75 Percentile',
        hoverinfo='none'
    ))
    # Mediana
    fig.add_trace(go.Scatter(
        x=anni_asse_x, y=median_data, mode='lines',
        name='Reddito Mediano (50°)',
        line={'width': 3, 'color': '#00B0F0'},
        hovertemplate='Anno %{x}<br>Reddito Mediano: €%{y:,.0f}<extra></extra>'
    ))

    fig.update_layout(
        title="Quale Sarà il Mio Tenore di Vita? (Reddito Annuo Reale)",
        xaxis_title="Anni",
        yaxis_title="Reddito Annuo Reale (€ di oggi)",
        yaxis_tickformat="€,d",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig

def plot_worst_scenarios_chart(data, patrimoni_finali, anni_totali):
    """Mostra un'analisi degli scenari peggiori (es. 10% dei casi)."""
    fig = go.Figure()
    
    # Identifica l'indice del 10° percentile dei patrimoni finali
    soglia_peggiore = np.percentile(patrimoni_finali, 10)
    indici_peggiori = np.where(patrimoni_finali <= soglia_peggiore)[0]
    
    anni_asse_x = np.linspace(0, anni_totali, data.shape[1])
    
    if len(indici_peggiori) > 0:
        # Mostra fino a un massimo di 50 linee per non affollare
        indici_da_mostrare = np.random.choice(indici_peggiori, size=min(50, len(indici_peggiori)), replace=False)
        
        for i in indici_da_mostrare:
            fig.add_trace(go.Scatter(
                x=anni_asse_x, y=data[i, :], mode='lines',
                line={'width': 1, 'color': 'rgba(255, 82, 82, 0.5)'},
                hoverinfo='none', showlegend=False
            ))

    # Aggiungi una linea mediana degli scenari peggiori per dare un riferimento
    mediana_scenari_peggiori = np.median(data[indici_peggiori, :], axis=0)
    fig.add_trace(go.Scatter(
        x=anni_asse_x, y=mediana_scenari_peggiori, mode='lines',
        name='Mediana Scenari Peggiori',
        line={'width': 2.5, 'color': '#FF5252'},
        hovertemplate='Anno %{x:.1f}<br>Patrimonio: €%{y:,.0f}<extra></extra>'
    ))
            
    fig.update_layout(
        title="Come si Comporta il Piano negli Scenari Peggiori? (Analisi del Rischio)",
        xaxis_title="Anni",
        yaxis_title="Patrimonio Reale (€ di oggi)",
        yaxis_tickformat="€,d",
        hovermode="x unified",
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )
    return fig

# --- Configurazione Pagina ---
st.set_page_config(
    page_title="Simulatore Finanziario Monte Carlo v2.0",
    page_icon="📈",
    layout="wide"
)

# Inizializza lo stato della sessione se non esiste
if 'simulazione_eseguita' not in st.session_state:
    st.session_state['simulazione_eseguita'] = False
    st.session_state['risultati'] = {}
    st.session_state['parametri'] = {}


# --- Dati di Default Portafoglio ---
def get_default_portfolio():
    return pd.DataFrame([
        {"Fondo": "Vanguard FTSE All-World UCITS ETF (USD) Accumulating", "Ticker": "VWCE", "Allocazione (%)": 90.0, "TER (%)": 0.22, "Rendimento Atteso (%)": 8.0, "Volatilità Attesa (%)": 15.0},
        {"Fondo": "Amundi Bloomberg Equal-Weight Commodity Ex-Agriculture", "Ticker": "CRB", "Allocazione (%)": 3.0, "TER (%)": 0.30, "Rendimento Atteso (%)": 5.0, "Volatilità Attesa (%)": 18.0},
        {"Fondo": "iShares MSCI EM UCITS ETF (Acc)", "Ticker": "EIMI", "Allocazione (%)": 3.0, "TER (%)": 0.18, "Rendimento Atteso (%)": 9.0, "Volatilità Attesa (%)": 22.0},
        {"Fondo": "Amundi MSCI Japan UCITS ETF Acc", "Ticker": "SJP", "Allocazione (%)": 3.0, "TER (%)": 0.12, "Rendimento Atteso (%)": 7.0, "Volatilità Attesa (%)": 16.0},
        {"Fondo": "iShares Automation & Robotics UCITS ETF", "Ticker": "RBOT", "Allocazione (%)": 1.0, "TER (%)": 0.40, "Rendimento Atteso (%)": 12.0, "Volatilità Attesa (%)": 25.0},
    ])

if 'portfolio' not in st.session_state:
    st.session_state.portfolio = get_default_portfolio()


# --- Titolo ---
st.title("📈 Simulatore di Pianificazione Finanziaria Monte Carlo v2.0")
st.markdown("Benvenuto nella versione 2.0 del simulatore. Utilizza i controlli nella barra laterale per configurare e lanciare la tua simulazione finanziaria.")

# --- Barra Laterale: Input Utente ---
st.sidebar.header("Configurazione Simulazione")

# --- Sezione Storico ---
with st.sidebar.expander("📚 Storico Simulazioni", expanded=False):
    saved_simulations = load_simulations()
    if not saved_simulations:
        st.caption("Nessuna simulazione salvata.")
    else:
        for sim in saved_simulations:
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.markdown(f"**{sim['name']}**")
                st.caption(f"Salvata il: {sim['timestamp']}")
            with col2:
                if st.button("📂", key=f"load_{sim['filename']}", help="Carica questa simulazione"):
                    data = load_simulation_data(sim['filename'])
                    st.session_state.parametri = data['parameters']
                    st.session_state.risultati = data['results']
                    st.session_state.simulazione_eseguita = True
                    st.rerun()
            with col3:
                if st.button("🗑️", key=f"delete_{sim['filename']}", help="Elimina questa simulazione"):
                    delete_simulation(sim['filename'])

# --- Sezione Parametri di Base ---
with st.sidebar.expander("1. Parametri di Base", expanded=True):
    p = st.session_state.get('parametri', {})
    
    # --- ETA' INIZIALE ---
    eta_iniziale = st.number_input("Età Iniziale", min_value=1, max_value=100, value=p.get('eta_iniziale', 27))
    with st.expander("ℹ️ Spiegazione: Età Iniziale"):
        st.markdown("""
        **Cosa rappresenta:** La tua età attuale. È il punto di partenza di tutta la simulazione.
        
        **Perché è importante:** Influenza direttamente l'orizzonte temporale sia della fase di accumulo che di quella di decumulo (pensione).
        """)

    # --- CAPITALE CONTO CORRENTE ---
    capitale_iniziale = st.number_input("Capitale Conto Corrente (€)", min_value=0, step=1000, value=p.get('capitale_iniziale', 17000))
    with st.expander("ℹ️ Spiegazione: Capitale Conto Corrente"):
        st.markdown("""
        **Cosa rappresenta:** La liquidità che possiedi oggi sul tuo conto corrente o in altri strumenti a bassissimo rischio (es. conti deposito).
        
        **Perché è importante:** È la tua prima linea di difesa contro le spese impreviste e la base da cui partono i tuoi investimenti futuri.
        """)

    # --- VALORE PORTAFOGLIO ETF ---
    etf_iniziale = st.number_input("Valore Portafoglio ETF (€)", min_value=0, step=1000, value=p.get('etf_iniziale', 600))
    with st.expander("ℹ️ Spiegazione: Valore Portafoglio ETF"):
        st.markdown("""
        **Cosa rappresenta:** Il valore totale attuale dei tuoi investimenti in ETF o altri strumenti finanziari che compongono il tuo portafoglio.
        
        **Perché è importante:** È il capitale che verrà messo al lavoro sui mercati e su cui verranno calcolati i rendimenti composti.
        """)
        
    # --- CONTRIBUTI MENSILI ---
    contributo_mensile_banca = st.number_input("Contributo Mensile Conto (€)", min_value=0, step=50, value=p.get('contributo_mensile_banca', 1300))
    contributo_mensile_etf = st.number_input("Contributo Mensile ETF (€)", min_value=0, step=50, value=p.get('contributo_mensile_etf', 300))
    with st.expander("ℹ️ Spiegazione: Contributi Mensili"):
        st.markdown("""
        **Cosa rappresenta:** La somma che riesci a risparmiare e investire ogni mese.
        
        **Perché è importante:** La costanza e l'entità dei tuoi contributi sono uno dei motori più potenti della crescita del tuo patrimonio nel tempo, quasi più importanti dei rendimenti stessi.
        """)
        
    # --- INFLAZIONE ---
    inflazione = st.slider("Inflazione Media Annua (%)", 0.0, 10.0, p.get('inflazione', 0.03) * 100, 0.1) / 100
    with st.expander("ℹ️ Spiegazione: Inflazione Media Annua"):
        st.markdown("""
        **Cosa rappresenta:** Il tasso a cui il costo della vita aumenta ogni anno. Un'inflazione del 3% significa che tra un anno, per comprare gli stessi beni e servizi, ti servirà il 3% in più di denaro.
        
        **Perché è importante:** È il "nemico silenzioso" dei tuoi risparmi. Il simulatore usa questo valore per calcolare il **potere d'acquisto reale** del tuo patrimonio in futuro. L'obiettivo è far crescere i tuoi investimenti a un tasso superiore all'inflazione.
        """)

    # --- ANNI ALL'INIZIO PRELIEVI ---
    anni_inizio_prelievo = st.number_input("Anni all'Inizio dei Prelievi", min_value=0, value=p.get('anni_inizio_prelievo', 35))
    with st.expander("ℹ️ Spiegazione: Anni all'Inizio dei Prelievi"):
        st.markdown("""
        **Cosa rappresenta:** Tra quanti anni prevedi di smettere di lavorare e iniziare a vivere del tuo patrimonio (la tua età di pensionamento sarà: Età Iniziale + questo valore).
        
        **Perché è importante:** Definisce la durata della fase di accumulo. Più questo numero è alto, più tempo avranno i tuoi investimenti per beneficiare dell'interesse composto.
        """)
        
    # --- PARAMETRI TECNICI ---
    n_simulazioni = st.slider("Numero Simulazioni", 10, 1000, p.get('n_simulazioni', 250), 10)
    anni_totali_input = st.number_input("Orizzonte Temporale (Anni)", min_value=1, max_value=100, value=p.get('anni_totali', 80))
    with st.expander("ℹ️ Spiegazione: Parametri Tecnici"):
        st.markdown("""
        - **Numero Simulazioni:** Quante "vite finanziarie" parallele calcolare. Un numero più alto (es. 500-1000) rende il risultato statisticamente più solido, ma richiede più tempo.
        - **Orizzonte Temporale:** La durata totale della simulazione. Assicurati che sia abbastanza lungo da coprire tutta la tua aspettativa di vita (es. Età Iniziale + Orizzonte > 95-100 anni).
        """)

# --- Sezione Portafoglio ETF ---
with st.sidebar.expander("2. Costruttore di Portafoglio ETF", expanded=True):
    p = st.session_state.get('parametri', {})
    st.write("Definisci il tuo portafoglio di investimenti:")

    # Carica i dati del portafoglio dallo stato della sessione o usa i default
    if 'etf_portfolio' not in st.session_state:
        st.session_state.etf_portfolio = p.get('etf_portfolio', pd.DataFrame([
            {'ETF': 'Azionario Globale', 'Allocazione (%)': 80, 'TER (%)': 0.22, 'Rendimento Atteso (%)': 8.0, 'Volatilità Attesa (%)': 15.0},
            {'ETF': 'Obbligazionario Globale', 'Allocazione (%)': 20, 'TER (%)': 0.18, 'Rendimento Atteso (%)': 2.5, 'Volatilità Attesa (%)': 5.0},
        ]))

    edited_df = st.data_editor(st.session_state.etf_portfolio, num_rows="dynamic", key="etf_editor")
    
    total_allocation = edited_df["Allocazione (%)"].sum()
    if not np.isclose(total_allocation, 100):
        st.warning(f"L'allocazione totale è {total_allocation:.2f}%. Assicurati che sia 100%.")
    else:
        st.success("Allocazione totale: 100%.")
    
    # Calcolo parametri aggregati del portafoglio per l'utente
    weights = edited_df["Allocazione (%)"] / 100
    rendimento_medio_portfolio = np.sum(weights * edited_df["Rendimento Atteso (%)"])
    volatilita_portfolio = np.sum(weights * edited_df["Volatilità Attesa (%)"])  # Semplificazione
    ter_etf_portfolio = np.sum(weights * edited_df["TER (%)"])

    st.markdown("---")
    st.markdown("##### Parametri Calcolati del Portafoglio:")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rendimento Medio Atteso", f"{rendimento_medio_portfolio:.2f}%")
    col2.metric("Volatilità Attesa", f"{volatilita_portfolio:.2f}%")
    col3.metric("TER Ponderato", f"{ter_etf_portfolio:.4f}%")
    st.caption("La volatilità aggregata è una media ponderata semplificata.")

    with st.expander("ℹ️ Spiegazione: Parametri del Portafoglio"):
        st.markdown("""
        - **Allocazione (%):** La percentuale del tuo capitale investita in ciascun ETF. La somma totale dovrebbe essere 100%.
        - **TER (%):** *Total Expense Ratio*. Il costo annuo di gestione dell'ETF, che viene sottratto direttamente dai rendimenti.
        - **Rendimento Atteso (%):** La tua stima del rendimento medio annuo *lordo* per quell'asset. Usa stime realistiche e conservative.
        - **Volatilità Attesa (%):** La misura delle oscillazioni di prezzo dell'asset. Una volatilità più alta significa un rischio maggiore e un "cono di incertezza" più ampio nei risultati.
        """)

# --- Sezione Strategie di Prelievo ---
with st.sidebar.expander("3. Strategie di Prelievo", expanded=False):
    p = st.session_state.get('parametri', {})
    
    # Inizializzazione pulita delle variabili
    prelievo_annuo = p.get('prelievo_annuo', 30000)
    percentuale_prelievo = p.get('percentuale_prelievo', 0.04)
    guardrail_superiore = p.get('guardrail_superiore', 0.20)
    guardrail_inferiore = p.get('guardrail_inferiore', 0.10)

    strategia_prelievo = st.selectbox(
        "Scegli la tua strategia di prelievo",
        options=['FISSO', 'REGOLA_4_PERCENTO', 'GUARDRAIL'],
        index=['FISSO', 'REGOLA_4_PERCENTO', 'GUARDRAIL'].index(p.get('strategia_prelievo', 'GUARDRAIL')),
    )
    with st.expander("ℹ️ Spiegazione: Strategie di Prelievo"):
        st.markdown("""
        - **FISSO:** Prelevi un importo fisso ogni anno, adeguato all'inflazione. Semplice ma rigido, può essere rischioso se i mercati vanno male all'inizio della pensione (*Sequence of Returns Risk*).
        - **REGOLA 4%:** Prelevi una percentuale fissa (es. 4%) del capitale *rimanente* all'inizio di ogni anno. È più flessibile e si adatta all'andamento del mercato.
        - **GUARDRAIL:** Una versione "intelligente" della Regola del 4%. I prelievi si adeguano all'inflazione, ma con dei "paraurti" (guardrail) che attivano correzioni se il tasso di prelievo diventa troppo alto o troppo basso, proteggendo il capitale e ottimizzando i flussi di cassa. **Consigliata per la maggior parte degli scenari.**
        """)

    if strategia_prelievo == 'FISSO':
        prelievo_annuo = st.number_input("Prelievo Annuo Lordo (€)", min_value=0, value=prelievo_annuo)
    elif strategia_prelievo == 'REGOLA_4_PERCENTO':
        percentuale_prelievo = st.slider("Percentuale di Prelievo (%)", 1.0, 10.0, percentuale_prelievo * 100, 0.1) / 100
    elif strategia_prelievo == 'GUARDRAIL':
        percentuale_prelievo = st.slider("Percentuale di Prelievo Iniziale (%)", 1.0, 10.0, percentuale_prelievo * 100, 0.1) / 100
        guardrail_superiore = st.slider("Guardrail Superiore (%)", 1.0, 20.0, guardrail_superiore * 100, 0.5) / 100
        guardrail_inferiore = st.slider("Guardrail Inferiore (%)", 1.0, 20.0, guardrail_inferiore * 100, 0.5) / 100
        with st.expander("ℹ️ Spiegazione: Parametri Guardrail"):
            st.markdown(f"""
            **Come funziona:**
            1.  Si calcola il prelievo dell'anno precedente, adeguato all'inflazione.
            2.  Si calcola il "Tasso di Prelievo Corrente" (prelievo / capitale attuale).
            3.  **Se** il tasso corrente supera il `tasso iniziale * (1 + Guardrail Superiore)` (es. {percentuale_prelievo:.1%} * {1+guardrail_superiore:.1%}), il prelievo viene **tagliato del 10%**.
            4.  **Se** il tasso corrente scende sotto il `tasso iniziale * (1 - Guardrail Inferiore)` (es. {percentuale_prelievo:.1%} * {1-guardrail_inferiore:.1%}), il prelievo viene **aumentato del 10%**.
            
            Questo protegge il capitale durante i crolli di mercato e ti permette di spendere di più durante i boom.
            """)
            
# --- Sezione Asset Allocation Dinamica ---
with st.sidebar.expander("4. Asset Allocation Dinamica (Glidepath)", expanded=False):
    p = st.session_state.get('parametri', {})
    
    use_glidepath = st.checkbox("Attiva Glidepath", value=p.get('use_glidepath', False))
    with st.expander("ℹ️ Spiegazione: Glidepath"):
        st.markdown("""
        **Cos'è:** Una strategia per ridurre automaticamente il rischio del portafoglio con l'avanzare dell'età.
        
        **Come funziona:** Sposta gradualmente il capitale da asset più rischiosi (ETF) a liquidità sicura (Conto Corrente) in un intervallo di anni da te definito.
        
        **Perché usarlo:** Per proteggere il capitale accumulato quando ti avvicini alla pensione e sei più vulnerabile alle fluttuazioni di mercato.
        """)
        
    if use_glidepath:
        start_glidepath_eta = st.slider("Età Inizio Glidepath", 40, 80, p.get('start_glidepath_eta', 55))
        end_glidepath_eta = st.slider("Età Fine Glidepath", 50, 90, p.get('end_glidepath_eta', 65))
        final_equity_percentage = st.slider("Percentuale Azionaria Finale (%)", 0, 100, p.get('final_equity_percentage', 60))
    else:
        start_glidepath_eta, end_glidepath_eta, final_equity_percentage = 0, 0, 0


# --- Sezione Tassazione e Costi ---
with st.sidebar.expander("5. Tassazione e Costi", expanded=False):
    p = st.session_state.get('parametri', {})
    tassazione_capital_gain = st.slider("Tassazione Capital Gain (%)", 0.0, 50.0, p.get('tassazione_capital_gain', 0.26) * 100, 0.5) / 100
    imposta_bollo_titoli = st.slider("Imposta di Bollo Titoli (%)", 0.0, 1.0, p.get('imposta_bollo_titoli', 0.002) * 100, 0.01) / 100
    imposta_bollo_liquidita = st.number_input("Imposta di Bollo Liquidità (€)", min_value=0, value=p.get('imposta_bollo_liquidita', 34))
    costo_fisso_deposito_titoli = st.number_input("Costo Fisso Deposito Titoli Annuo (€)", min_value=0, value=p.get('costo_fisso_deposito_titoli', 0))

    with st.expander("ℹ️ Spiegazione: Tasse e Costi"):
        st.markdown("""
        - **Tassazione Capital Gain:** L'aliquota applicata alle plusvalenze realizzate quando vendi un ETF in profitto.
        - **Imposta di Bollo Titoli:** La tassa patrimoniale annuale sul valore del tuo portafoglio titoli (tipicamente 0.20%).
        - **Imposta di Bollo Liquidità:** La tassa fissa annuale sul conto corrente se la giacenza media supera i 5.000€.
        - **Costo Fisso Deposito Titoli:** Eventuali costi amministrativi annuali applicati dal tuo broker.
        """)

# --- Sezione Fondo Pensione ---
with st.sidebar.expander("6. Fondo Pensione", expanded=False):
    p = st.session_state.get('parametri', {})
    usa_fp = st.checkbox("Attiva simulazione Fondo Pensione", value=p.get('usa_fp', False))
    if usa_fp:
        fp_capitale_iniziale = st.number_input("Capitale Iniziale FP (€)", min_value=0, value=p.get('fp_capitale_iniziale', 20000))
        fp_contributo_mensile = st.number_input("Contributo Mensile FP (€)", min_value=0, value=p.get('fp_contributo_mensile', 200))
        fp_eta_liquidazione = st.number_input("Età Liquidazione FP", min_value=50, value=p.get('fp_eta_liquidazione', 67))
        fp_rendimento_netto = st.slider("Rendimento Annuo Netto FP (%)", 0.0, 10.0, p.get('fp_rendimento_netto', 0.04) * 100, 0.1) / 100
        fp_perc_liquidazione_capitale = st.slider("Percentuale Ritiro Capitale (%)", 0, 100, p.get('fp_perc_liquidazione_capitale', 50))
        fp_aliquota_tassazione_finale = st.slider("Aliquota Tassazione Finale FP (%)", 9.0, 23.0, p.get('fp_aliquota_tassazione_finale', 0.15) * 100, 0.5) / 100
        
        with st.expander("ℹ️ Spiegazione: Fondo Pensione"):
            st.markdown("""
            **Come funziona:**
            1.  Simula un piano pensionistico complementare separato dai tuoi investimenti principali.
            2.  I contributi crescono con un **rendimento netto** (già al netto dei costi di gestione e della tassazione sui rendimenti del 20%).
            3.  All'**Età di Liquidazione**, puoi ritirare una parte del capitale (fino al 50% per legge) pagando un'**aliquota finale agevolata** (dal 15% al 9%).
            4.  Il capitale rimanente viene convertito in una rendita annua che si aggiunge alle tue entrate.
            """)
    else:
        fp_capitale_iniziale, fp_contributo_mensile, fp_eta_liquidazione, fp_rendimento_netto, fp_perc_liquidazione_capitale, fp_aliquota_tassazione_finale = 0, 0, 0, 0, 0, 0

# --- Sezione Altre Entrate ---
with st.sidebar.expander("7. Altre Entrate", expanded=False):
    p = st.session_state.get('parametri', {})
    usa_pensione_pubblica = st.checkbox("Attiva Pensione Pubblica", value=p.get('usa_pensione_pubblica', False))
    if usa_pensione_pubblica:
        pensione_pubblica_importo_annuo = st.number_input("Importo Annuo Pensione Pubblica (€)", min_value=0, value=p.get('pensione_pubblica_importo_annuo', 15000))
        eta_inizio_pensione_pubblica = st.number_input("Età Inizio Pensione Pubblica", min_value=50, value=p.get('eta_inizio_pensione_pubblica', 67))
        
        with st.expander("ℹ️ Spiegazione: Altre Entrate"):
            st.markdown("""
            Usa questa sezione per includere altre fonti di reddito che riceverai in futuro, come la pensione INPS, affitti da immobili, ecc. Questi importi verranno aggiunti al tuo conto corrente ogni anno, adeguati all'inflazione.
            """)
    else:
        pensione_pubblica_importo_annuo, eta_inizio_pensione_pubblica = 0, 0

# --- Pulsante Esecuzione ---
if st.sidebar.button("🚀 Esegui Simulazione", type="primary"):
    # Validazione allocazione
    if not np.isclose(st.session_state.etf_portfolio["Allocazione (%)"].sum(), 100):
        st.sidebar.error("L'allocazione del portafoglio deve essere esattamente 100% per eseguire la simulazione.")
    else:
        st.session_state.parametri = {
            'eta_iniziale': eta_iniziale, 'capitale_iniziale': capitale_iniziale, 'etf_iniziale': etf_iniziale,
            'contributo_mensile_banca': contributo_mensile_banca, 'contributo_mensile_etf': contributo_mensile_etf, 
            'etf_portfolio': st.session_state.etf_portfolio,
            'inflazione': inflazione, 'anni_inizio_prelievo': anni_inizio_prelievo,
            'prelievo_annuo': prelievo_annuo, 'n_simulazioni': n_simulazioni, 'anni_totali': anni_totali_input,
            'strategia_prelievo': strategia_prelievo, 'percentuale_prelievo': percentuale_prelievo,
            'guardrail_superiore': guardrail_superiore, 'guardrail_inferiore': guardrail_inferiore,
            'use_glidepath': use_glidepath, 'start_glidepath_eta': start_glidepath_eta, 'end_glidepath_eta': end_glidepath_eta,
            'final_equity_percentage': final_equity_percentage,
            'tassazione_capital_gain': tassazione_capital_gain, 'imposta_bollo_titoli': imposta_bollo_titoli,
            'imposta_bollo_liquidita': imposta_bollo_liquidita, 'costo_fisso_deposito_titoli': costo_fisso_deposito_titoli,
            'usa_fp': usa_fp, 'fp_capitale_iniziale': fp_capitale_iniziale, 'fp_contributo_mensile': fp_contributo_mensile,
            'fp_eta_liquidazione': fp_eta_liquidazione, 'fp_rendimento_netto': fp_rendimento_netto,
            'fp_perc_liquidazione_capitale': fp_perc_liquidazione_capitale, 'fp_aliquota_tassazione_finale': fp_aliquota_tassazione_finale,
            'usa_pensione_pubblica': usa_pensione_pubblica, 'pensione_pubblica_importo_annuo': pensione_pubblica_importo_annuo, 'eta_inizio_pensione_pubblica': eta_inizio_pensione_pubblica
        }

        with st.spinner("Esecuzione della simulazione Monte Carlo... Questo potrebbe richiedere alcuni istanti."):
            try:
                st.session_state.risultati = engine.run_full_simulation(st.session_state.parametri)
                st.session_state.simulazione_eseguita = True
                st.rerun() # Ricarica l'app per mostrare i risultati
            except ValueError as e:
                st.error(f"Errore nei parametri: {e}")
            except Exception as e:
                st.error(f"Si è verificato un errore inaspettato durante la simulazione: {e}")


# --- Area Principale ---
st.markdown("---")


if not st.session_state.simulazione_eseguita:
    st.header("Risultati della Simulazione")
    st.info("I risultati appariranno qui dopo aver eseguito la simulazione.")
else:
    stats = st.session_state.risultati['statistiche']
    params = st.session_state.parametri

    # --- Sezione Salvataggio ---
    with st.expander("💾 Salva Risultati Simulazione"):
        simulation_name = st.text_input("Dai un nome a questa simulazione", f"Simulazione del {datetime.now().strftime('%d/%m/%Y %H:%M')}")
        if st.button("Salva Simulazione"):
            save_simulation(simulation_name, params, st.session_state.risultati)

    # --- Cruscotto Strategico ---
    st.header("Cruscotto Strategico Riepilogativo")
    
    # Definisco stats_prelievi prima di usarlo
    stats_prelievi = st.session_state.risultati['statistiche_prelievi']
    
    prob_successo = 1 - stats['probabilita_fallimento']
    tenore_vita_mediano = stats_prelievi['totale_reale_medio_annuo']
    patrimonio_reale_finale = stats['patrimonio_finale_mediano_reale']

    def get_color_for_probability(p):
        if p >= 0.85: return "green"
        if p >= 0.70: return "orange"
        return "red"

    col1, col2, col3 = st.columns(3)
    with col1:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob_successo * 100,
            title = {'text': "Probabilità di Successo"},
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': get_color_for_probability(prob_successo)},
                'steps' : [
                    {'range': [0, 70], 'color': 'rgba(255, 0, 0, 0.1)'},
                    {'range': [70, 85], 'color': 'rgba(255, 165, 0, 0.1)'},
                    {'range': [85, 100], 'color': 'rgba(0, 128, 0, 0.1)'}],
                'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 85}
            }
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure(go.Indicator(
            mode = "number",
            value = tenore_vita_mediano,
            title = {"text": "Tenore di Vita Annuo<br>(Reale Mediano)"},
            number = {'prefix': "€", 'valueformat': ',.0f'}
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        fig = go.Figure(go.Indicator(
            mode = "number",
            value = patrimonio_reale_finale,
            title = {"text": "Patrimonio Finale<br>(Reale Mediano)"},
            number = {'prefix': "€", 'valueformat': ',.0f'}
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)


    # --- Riepilogo Statistico ---
    st.header("Riepilogo Statistico Chiave")

    st.write("---")
    st.markdown("##### Capitale Nominale")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Patrimonio Iniziale", f"€ {stats['patrimonio_iniziale']:,.0f}",
        help="La somma del capitale che hai all'inizio della simulazione."
    )
    col2.metric(
        "Patrimonio Finale Mediano (50°)", f"€ {stats['patrimonio_finale_mediano_nominale']:,.0f}",
        help="Il risultato che si trova esattamente nel mezzo di tutti gli scenari. È la stima più realistica."
    )
    col3.metric(
        "Patrimonio Finale (Top 10% - 90°)", f"€ {stats['patrimonio_finale_top_10_nominale']:,.0f}",
        help="Lo scenario 'da sogno'. C'è solo un 10% di probabilità che le cose vadano meglio di così."
    )
    col4.metric(
        "Patrimonio Finale (Peggior 10% - 10°)", f"€ {stats['patrimonio_finale_peggior_10_nominale']:,.0f}",
        help="Lo scenario 'notte insonne'. C'è un 10% di probabilità che le cose vadano peggio di così."
    )

    st.write("---")
    st.markdown("##### Capitale Reale & Rischio")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Patrimonio Reale Finale Mediano (50°)", f"€ {stats['patrimonio_finale_mediano_reale']:,.0f}",
        help="Il potere d'acquisto mediano del tuo patrimonio a fine piano, espresso in Euro di oggi. La metrica più importante."
    )
    col2.metric(
        "Probabilità di Fallimento", f"{stats['probabilita_fallimento']:.2%}",
        delta=f"{-stats['probabilita_fallimento']:.2%}", delta_color="inverse",
        help="La probabilità di finire i soldi prima della fine della simulazione."
    )
    col3.metric(
        "Drawdown Massimo Peggiore", f"{stats['drawdown_massimo_peggiore']:.2%}",
        delta=f"{stats['drawdown_massimo_peggiore']:.2%}", delta_color="inverse",
        help="La perdita percentuale più grande dal picco, nello scenario peggiore. Misura il 'dolore' massimo che potresti sopportare."
    )
    col4.metric(
        "Sharpe Ratio Medio", f"{stats['sharpe_ratio_medio']:.2f}",
        help="Il rendimento ottenuto per ogni unità di rischio. Un valore più alto è meglio (sopra 1 è ottimo)."
    )

    # --- Riepilogo Entrate in Pensione ---
    st.write("---")
    st.header("Riepilogo Entrate in Pensione (Valori Reali Medi)")
    st.markdown("Queste metriche mostrano il tenore di vita **medio annuo** che puoi aspettarti durante la fase di ritiro, espresso nel potere d'acquisto di oggi.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Prelievo Medio dal Patrimonio", f"€ {stats_prelievi['prelievo_reale_medio']:,.0f}",
        help="L'importo medio annuo prelevato dal tuo portafoglio (ETF+Liquidità) durante la pensione, in potere d'acquisto di oggi."
    )
    col2.metric(
        "Pensione Pubblica Annua", f"€ {stats_prelievi['pensione_pubblica_reale_annua']:,.0f}",
        help="L'importo della pensione pubblica che hai inserito, in potere d'acquisto di oggi."
    )
    col3.metric(
        "Rendita Media da Fondo Pensione", f"€ {stats_prelievi['rendita_fp_reale_media']:,.0f}",
        help="La rendita annua media generata dal capitale del tuo fondo pensione, in potere d'acquisto di oggi."
    )
    col4.metric(
        "TOTALE ENTRATE MEDIE ANNUE", f"€ {stats_prelievi['totale_reale_medio_annuo']:,.0f}",
        help="La somma di tutte le entrate medie in pensione. Misura il tuo tenore di vita medio annuo una volta in ritiro.",
    )


    # --- Grafici di Simulazione ---
    st.write("---")
    st.header("Analisi Dettagliata per Fasi")

    tab_accumulo, tab_decumulo, tab_dettaglio = st.tabs([
        "📊 Fase di Accumulo", 
        "🏖️ Fase di Decumulo (Pensione)",
        "🧾 Dettaglio Flussi di Cassa (Mediano)"
    ])

    with tab_accumulo:
        eta_pensionamento = params['eta_iniziale'] + params['anni_inizio_prelievo']
        st.subheader(f"Dall'età attuale ({params['eta_iniziale']} anni) fino alla pensione (a {eta_pensionamento} anni)")
        st.markdown("In questa fase, i tuoi sforzi si concentrano sulla **costruzione del patrimonio**. I tuoi contributi mensili e annuali, uniti ai rendimenti composti degli investimenti, lavorano insieme per far crescere il capitale che ti sosterrà in futuro.")
        st.markdown("---")
        
        dati_grafici = st.session_state.risultati['dati_grafici_principali']
        
        fig_reale = plot_percentile_chart(
            dati_grafici['reale'], 'Evoluzione Patrimonio Reale (Tutti gli Scenari)', 'Patrimonio Reale (€)', 
            color_median='#C00000', color_fill='#C00000',
            anni_totali=params['anni_totali']
        )
        fig_reale.add_vline(x=params['anni_inizio_prelievo'], line_width=2, line_dash="dash", line_color="grey", annotation_text="Inizio Prelievi")
        st.plotly_chart(fig_reale, use_container_width=True)
        st.markdown("<div style='text-align: center; font-size: 0.9em; font-style: italic;'>Questo è il grafico della verità. Tiene conto dell'inflazione, mostrando il vero potere d'acquisto. La linea rossa è lo scenario mediano (50° percentile), le aree colorate mostrano i range di probabilità (dal 10° al 90° percentile).</div>", unsafe_allow_html=True)

        fig_nominale = plot_percentile_chart(
            dati_grafici['nominale'], 'Evoluzione Patrimonio Nominale (Tutti gli Scenari)', 'Patrimonio (€)',
            color_median='#4472C4', color_fill='#4472C4',
            anni_totali=params['anni_totali']
        )
        fig_nominale.add_vline(x=params['anni_inizio_prelievo'], line_width=2, line_dash="dash", line_color="grey", annotation_text="Inizio Prelievi")
        st.plotly_chart(fig_nominale, use_container_width=True)
        st.markdown("<div style='text-align: center; font-size: 0.9em; font-style: italic;'>Questo grafico mostra il valore 'nominale', cioè quanti Euro vedrai scritti sul tuo estratto conto in futuro, senza considerare l'inflazione.</div>", unsafe_allow_html=True)


    with tab_decumulo:
        # Calcolo dinamico delle età
        eta_pensionamento = params['eta_iniziale'] + params['anni_inizio_prelievo']
        eta_pensione_pubblica = params['eta_iniziale'] + params['eta_inizio_pensione_pubblica']
        eta_ritiro_fp = params['fp_eta_liquidazione']

        st.subheader(f"Dalla pensione (a {eta_pensionamento} anni) in poi")
        
        testo_decumulo = f"""
        A partire da **{eta_pensionamento} anni**, smetti di versare e inizi a **prelevare dal tuo patrimonio** per sostenere il tuo tenore di vita. 
        A questo si aggiungeranno le altre fonti di reddito che hai configurato:
        - La **pensione pubblica** a partire da **{eta_pensione_pubblica} anni**.
        """
        if params['usa_fp']:
            testo_decumulo += f"\n- L'eventuale **rendita del fondo pensione** a partire da **{eta_ritiro_fp} anni**."
        
        testo_decumulo += "\n\nL'obiettivo è far sì che il patrimonio duri per tutto l'orizzonte temporale desiderato."
        st.markdown(testo_decumulo)
        st.markdown("---")

        dati_principali = st.session_state.risultati['dati_grafici_principali']

        # --- Grafico 1: Cono del Reddito ---
        st.subheader("📊 Quale sarà il mio tenore di vita in pensione?")
        st.markdown("""
        Questo grafico è forse il più importante per la tua pianificazione. Non ti mostra solo un numero, ma **l'intera gamma dei possibili redditi annuali reali** (cioè, il potere d'acquisto di oggi) che potrai avere durante la pensione.

        - **La linea blu centrale (Mediana):** È lo scenario più probabile, il tuo obiettivo realistico.
        - **L'area azzurra (25°-75° percentile):** È l'intervallo di reddito 'plausibile'. C'è una buona probabilità che il tuo tenore di vita rientri in questa fascia.
        - **L'area più chiara (10°-90° percentile):** Rappresenta gli estremi. La parte alta è lo scenario da sogno, la parte bassa è lo scenario più pessimistico ma comunque possibile. Usala per capire quale potrebbe essere il tuo reddito minimo in caso di andamenti di mercato molto sfortunati.
        """)
        fig_reddito = plot_income_cone_chart(
            dati_principali['reddito_reale_annuo'], 
            params['anni_totali'],
            params['anni_inizio_prelievo']
        )
        st.plotly_chart(fig_reddito, use_container_width=True)
        st.markdown("---")

        # --- Grafico 2: Analisi Scenari Peggiori ---
        st.subheader("🔥 Il piano sopravviverà a una crisi di mercato iniziale?")
        st.markdown("""
        Questo grafico è il *crash test* del tuo piano pensionistico. Affronta la paura più grande di ogni pensionato: il **Rischio da Sequenza dei Rendimenti**. In parole semplici, una forte crisi di mercato **proprio all'inizio della pensione** è molto più dannosa di una che avviene 20 anni dopo, perché erode il capitale da cui stai iniziando a prelevare, dandogli meno tempo per recuperare.

        Qui isoliamo il **10% degli scenari più sfortunati** della simulazione. Ogni linea rossa sottile rappresenta l'evoluzione del patrimonio in uno di questi futuri avversi.
        
        - **Cosa osservare:** Il tuo piano è robusto se, anche in questi scenari, il patrimonio non si azzera troppo in fretta. Se vedi che molte linee crollano a zero rapidamente, potrebbe essere un segnale che il tuo piano è troppo aggressivo o la tua percentuale di prelievo troppo alta per resistere a una "tempesta perfetta" iniziale.
        """)
        fig_worst = plot_worst_scenarios_chart(
            dati_principali['reale'],
            st.session_state.risultati['statistiche']['patrimoni_reali_finali'],
            params['anni_totali']
        )
        fig_worst.add_vline(x=params['anni_inizio_prelievo'], line_width=2, line_dash="dash", line_color="grey", annotation_text="Inizio Prelievi")
        st.plotly_chart(fig_worst, use_container_width=True)
    
    with tab_dettaglio:
        st.subheader("Analisi Finanziaria Annuale Dettagliata (Simulazione Mediana)")
        st.markdown("Questa tabella è la 'radiografia' dello scenario mediano (il più probabile). Mostra, anno per anno, tutti i flussi finanziari.")
        
        dati_tabella = st.session_state.risultati['dati_grafici_avanzati']['dati_mediana']
        
        # Costruzione del DataFrame
        num_anni = params['anni_totali']
        df = pd.DataFrame({
            'Anno': np.arange(1, num_anni + 1),
            'Età': params['eta_iniziale'] + np.arange(num_anni),
            'Obiettivo Prelievo (Nom.)': dati_tabella['prelievi_target_nominali'][:num_anni],
            'Prelievo Effettivo (Nom.)': dati_tabella['prelievi_effettivi_nominali'][:num_anni],
            'Fonte: Conto Corrente': dati_tabella['prelievi_da_banca_nominali'][:num_anni],
            'Fonte: Vendita ETF': dati_tabella['prelievi_da_etf_nominali'][:num_anni],
            'Vendita ETF (Rebalance)': dati_tabella['vendite_rebalance_nominali'][:num_anni],
            'Prelievo Effettivo (Reale)': dati_tabella['prelievi_effettivi_reali'][:num_anni],
            'Pensione Pubblica (Reale)': dati_tabella['pensioni_pubbliche_reali'][:num_anni],
            'Rendita FP (Reale)': dati_tabella['rendite_fp_reali'][:num_anni],
            'Liquidazione Capitale FP (Nom.)': dati_tabella['fp_liquidato_nominale'][:num_anni],
            'Saldo Conto Fine Anno (Reale)': dati_tabella['saldo_banca_reale'][:num_anni],
            'Valore ETF Fine Anno (Reale)': dati_tabella['saldo_etf_reale'][:num_anni]
        })

        # Aggiungo la colonna calcolata
        df['Entrate Anno (Reali)'] = df['Prelievo Effettivo (Reale)'] + df['Pensione Pubblica (Reale)'] + df['Rendita FP (Reale)']
        
        # Riorganizzo le colonne per la visualizzazione
        colonne_visualizzate = [
            'Anno', 'Età', 'Obiettivo Prelievo (Nom.)', 'Prelievo Effettivo (Nom.)', 
            'Fonte: Conto Corrente', 'Fonte: Vendita ETF', 'Vendita ETF (Rebalance)', 
            'Liquidazione Capitale FP', 'Prelievo Effettivo (Reale)', 'Pensione Pubblica (Reale)', 
            'Rendita FP (Reale)', 'Entrate Anno (Reali)', 'Saldo Conto Fine Anno (Reale)', 
            'Valore ETF Fine Anno (Reale)'
        ]
        
        st.dataframe(
            df[colonne_visualizzate],
            height=500,
            column_config={
                "Obiettivo Prelievo (Nom.)": st.column_config.NumberColumn(format="€ %.0f"),
                "Prelievo Effettivo (Nom.)": st.column_config.NumberColumn(format="€ %.0f"),
                "Fonte: Conto Corrente": st.column_config.NumberColumn(format="€ %.0f"),
                "Fonte: Vendita ETF": st.column_config.NumberColumn(format="€ %.0f"),
                "Vendita ETF (Rebalance)": st.column_config.NumberColumn(format="€ %.0f"),
                "Liquidazione Capitale FP": st.column_config.NumberColumn(format="€ %.0f"),
                "Prelievo Effettivo (Reale)": st.column_config.NumberColumn(format="€ %.0f"),
                "Pensione Pubblica (Reale)": st.column_config.NumberColumn(format="€ %.0f"),
                "Rendita FP (Reale)": st.column_config.NumberColumn(format="€ %.0f"),
                "Entrate Anno (Reali)": st.column_config.NumberColumn(format="€ %.0f", help="La somma di tutte le entrate reali. Questa cifra misura il tuo vero tenore di vita annuale."),
                "Saldo Conto Fine Anno (Reale)": st.column_config.NumberColumn(format="€ %.0f"),
                "Valore ETF Fine Anno (Reale)": st.column_config.NumberColumn(format="€ %.0f"),
            }
        )

        with st.expander("Guida alla Lettura della Tabella"):
            st.markdown("""
            - **Obiettivo Prelievo vs Prelievo Effettivo**: L''Obiettivo' è quanto vorresti prelevare. L''Effettivo' è quanto prelevi realmente. Se hai pochi soldi, l''Effettivo' sarà più basso.
            - **Fonte Conto vs Fonte ETF**: Mostrano da dove provengono i soldi per il prelievo. Prima si usa la liquidità sul conto, poi si vendono gli ETF.
            - **Vendita ETF (Rebalance)**: NON sono soldi spesi. Sono vendite fatte per ridurre il rischio (seguendo il Glidepath). I soldi vengono spostati da ETF a liquidità.
            - **Liquidazione Capitale FP**: Somma che ricevi tutta in una volta dal fondo pensione all'età scelta. Aumenta di molto la tua liquidità in quell'anno.
            - **Entrate Anno (Reali)**: La somma di tutte le tue entrate (prelievi, pensioni) in potere d'acquisto di oggi. Questa cifra misura il tuo vero tenore di vita annuale.
            """) 