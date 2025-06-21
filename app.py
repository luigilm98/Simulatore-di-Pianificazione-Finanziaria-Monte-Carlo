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

def plot_wealth_composition_chart(initial, contributions, gains):
    """Crea un grafico a barre per la composizione della ricchezza finale."""
    labels = ['Patrimonio Iniziale', 'Contributi Versati', 'Guadagni da Investimento']
    values = [initial, contributions, gains]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Blu, Arancione, Verde

    fig = go.Figure(data=[go.Bar(
        x=labels, 
        y=values,
        marker_color=colors,
        text=[f"€{v:,.0f}" for v in values],
        textposition='auto'
    )])
    
    fig.update_layout(
        title_text='Da Dove Viene la Tua Ricchezza? (Patrimonio Nominale Mediano)',
        yaxis_title_text='Euro (€)',
        xaxis_title_text='Fonte del Patrimonio',
        bargap=0.4,
        yaxis_tickformat="€,d"
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
    eta_iniziale = st.number_input("Età Iniziale", min_value=1, max_value=100, value=p.get('eta_iniziale', 27), help="La tua età attuale. È il punto di partenza per tutti i calcoli temporali.")
    capitale_iniziale = st.number_input("Capitale Conto Corrente (€)", min_value=0, step=1000, value=p.get('capitale_iniziale', 17000), help="La liquidità che hai oggi sul conto corrente o in asset a bassissimo rischio/rendimento.")
    etf_iniziale = st.number_input("Valore Portafoglio ETF (€)", min_value=0, step=1000, value=p.get('etf_iniziale', 600), help="Il valore di mercato attuale di tutti i tuoi investimenti in ETF/azioni.")
    contributo_mensile_banca = st.number_input("Contributo Mensile Conto (€)", min_value=0, step=50, value=p.get('contributo_mensile_banca', 1300), help="La cifra che riesci a risparmiare e accantonare sul conto corrente ogni mese. Questi soldi verranno usati per il ribilanciamento o per le spese.")
    contributo_mensile_etf = st.number_input("Contributo Mensile ETF (€)", min_value=0, step=50, value=p.get('contributo_mensile_etf', 300), help="La cifra che investi attivamente ogni mese nel tuo portafoglio ETF. Questo è il motore principale del tuo Piano di Accumulo (PAC).")
    inflazione = st.slider("Inflazione Media Annua (%)", 0.0, 10.0, p.get('inflazione', 0.03) * 100, 0.1, help="Il tasso a cui i prezzi aumentano e il denaro perde potere d'acquisto. Un'inflazione del 3% significa che tra un anno, 100€ compreranno beni per 97€.") / 100
    anni_inizio_prelievo = st.number_input("Anni all'Inizio dei Prelievi", min_value=0, value=p.get('anni_inizio_prelievo', 35), help="Tra quanti anni prevedi di smettere di lavorare e iniziare a vivere del tuo patrimonio (e pensione). Questo segna il passaggio dalla fase di Accumulo a quella di Decumulo.")
    n_simulazioni = st.slider("Numero Simulazioni", 10, 1000, p.get('n_simulazioni', 250), 10, help="Più simulazioni esegui, più accurata sarà la stima delle probabilità. 250 è un buon compromesso tra velocità e precisione.")
    anni_totali_input = st.number_input("Orizzonte Temporale (Anni)", min_value=1, max_value=100, value=p.get('anni_totali', 80), help="La durata totale della simulazione. Assicurati che sia abbastanza lunga da coprire tutta la tua aspettativa di vita.")

# --- Sezione Portafoglio ETF ---
with st.sidebar.expander("2. Costruttore di Portafoglio ETF", expanded=True):
    st.markdown("""
    **Definisci qui la composizione del tuo portafoglio di investimenti.**
    - **Allocazione (%):** La percentuale di ogni strumento sul totale. La somma deve fare 100%.
    - **Rendimento Atteso Annuo (%):** La stima del rendimento medio annuo per ogni ETF, al lordo dei costi.
    - **Volatilità Annuo (%):** La deviazione standard dei rendimenti. Misura il rischio dello strumento. Valori più alti indicano maggiori oscillazioni di prezzo.
    - **Costo Annuo (TER) (%):** Il costo totale che l'emittente dell'ETF addebita annualmente per la gestione.
    """)
    
    # Recupera il portafoglio di default o quello in sessione
    portfolio_key = 'etf_portfolio'
    edited_portfolio = st.data_editor(
        st.session_state.portfolio,
        column_config={
            "Allocazione (%)": st.column_config.NumberColumn(format="%.2f%%", min_value=0, max_value=100),
            "TER (%)": st.column_config.NumberColumn(format="%.2f%%", min_value=0),
            "Rendimento Atteso (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "Volatilità Attesa (%)": st.column_config.NumberColumn(format="%.2f%%"),
        },
        num_rows="dynamic",
        key=portfolio_key
    )

    total_allocation = edited_portfolio["Allocazione (%)"].sum()
    if not np.isclose(total_allocation, 100):
        st.warning(f"L'allocazione totale è {total_allocation:.2f}%. Assicurati che sia 100%.")
    else:
        st.success("Allocazione totale: 100%.")
    
    st.session_state.portfolio = edited_portfolio
    
    # Calcolo dei parametri aggregati dal portafoglio
    weights = edited_portfolio["Allocazione (%)"] / 100
    rendimento_medio_portfolio = np.sum(weights * edited_portfolio["Rendimento Atteso (%)"]) / 100
    volatilita_portfolio = np.sum(weights * edited_portfolio["Volatilità Attesa (%)"]) / 100 # Semplificazione, in realtà servirebbe la matrice di correlazione
    ter_etf_portfolio = np.sum(weights * edited_portfolio["TER (%)"]) / 100

    st.markdown("---")
    st.markdown("##### Parametri Calcolati dal Portafoglio:")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rendimento Medio", f"{rendimento_medio_portfolio:.2%}")
    col2.metric("Volatilità Attesa", f"{volatilita_portfolio:.2%}")
    col3.metric("TER Ponderato", f"{ter_etf_portfolio:.4%}")
    st.caption("La volatilità aggregata è una media ponderata semplificata.")

# --- Sezione Tassazione ---
with st.sidebar.expander("3. Tassazione"):
    p = st.session_state.get('parametri', {})
    tassazione_capital_gain = st.slider(
        "Tassazione Capital Gain (%)", 0.0, 50.0, p.get('tassazione_capital_gain', 0.26) * 100, 1.0, 
        help="**Quale aliquota si applica ai profitti?** Imposta la tassazione sui guadagni in conto capitale (la differenza tra prezzo di vendita e di acquisto). In Italia, l'aliquota standard per gli strumenti finanziari è del 26%."
    ) / 100

# --- Sezione Strategie di Prelievo ---
with st.sidebar.expander("4. Strategie di Prelievo", expanded=True):
    p = st.session_state.get('parametri', {})
    strategia_prelievo = st.selectbox(
        "Strategia di Prelievo",
        options=['FISSO', 'REGOLA_4_PERCENTO', 'GUARDRAIL'],
        index=['FISSO', 'REGOLA_4_PERCENTO', 'GUARDRAIL'].index(p.get('strategia_prelievo', 'REGOLA_4_PERCENTO')),
        help="""
        **Come vuoi prelevare i soldi in pensione?**
        - **FISSO:** Prelevi ogni anno un importo fisso (rivalutato per l'inflazione), definito da te. Semplice ma rigido.
        - **REGOLA 4%:** Ogni anno prelevi una percentuale fissa (es. 4%) del **capitale residuo**. Il prelievo si adatta all'andamento del mercato (più alto se il mercato sale, più basso se scende).
        - **GUARDRAIL:** Una versione avanzata della regola del 4%. Definisci un prelievo target, ma questo viene aggiustato verso l'alto o il basso solo se il portafoglio supera delle "barriere" (guardrail) predefinite. Protegge dai prelievi eccessivi durante i crolli e permette di beneficiare dei rialzi.
        """
    )
    prelievo_annuo = st.number_input(
        "Importo Prelievo Fisso Annuo (€)",
        min_value=0, step=500, value=p.get('prelievo_annuo', 12000),
        help="**Usato SOLO con la strategia 'FISSO'.** Imposta l'esatto importo lordo che vuoi prelevare il primo anno di pensione. Gli anni successivi, questo importo verrà adeguato all'inflazione."
    )
    percentuale_regola_4 = st.slider(
        "Percentuale Regola 4% / Prelievo Iniziale (%)", 0.0, 10.0, p.get('percentuale_regola_4', 0.04) * 100, 0.1,
        help="**Tasso di prelievo per le strategie 'REGOLA 4%' e 'GUARDRAIL'.** La 'regola del 4%' è uno standard basato su studi storici, ma puoi adattarla. È la percentuale del patrimonio che prelevi il primo anno."
    ) / 100
    banda_guardrail = st.slider(
        "Banda Guardrail (%)", 0.0, 50.0, p.get('banda_guardrail', 0.10) * 100, 1.0,
        help="**Solo per 'GUARDRAIL'.** Definisce le barriere. Esempio: con 10%, se il prelievo calcolato è il 10% più alto o più basso di quello dell'anno precedente, viene 'tagliato' per evitare scossoni eccessivi."
    ) / 100

# --- Sezione Asset Allocation Dinamica (Glidepath) ---
with st.sidebar.expander("5. Asset Allocation Dinamica (Glidepath)"):
    p = st.session_state.get('parametri', {})
    attiva_glidepath = st.checkbox(
        "Attiva Glidepath (Ribilanciamento Automatico)", 
        value=p.get('attiva_glidepath', False),
        help="**Vuoi ridurre il rischio con l'avvicinarsi della pensione?** Se attivato, il glidepath riduce gradualmente l'esposizione azionaria (più rischiosa) a favore di quella obbligazionaria (più sicura) man mano che ti avvicini all'età del ritiro."
    )
    anni_glidepath = st.number_input(
        "Anni di Durata Glidepath", min_value=1, value=p.get('anni_glidepath', 20), disabled=not attiva_glidepath,
        help="**In quanti anni vuoi completare la transizione?** Definisce il periodo prima della pensione in cui inizia il ribilanciamento. Es: 20 anni significa che la transizione inizia 20 anni prima del ritiro."
    )
    allocazione_finale_obbligazionario = st.slider(
        "Allocazione Finale Obbligazionario (%)", 0, 100, p.get('allocazione_finale_obbligazionario', 60), disabled=not attiva_glidepath,
        help="**Qual è l'asset allocation target alla fine del percorso?** Indica la percentuale di portafoglio che sarà investita in obbligazionario (bond) al momento della pensione."
    )

# --- Sezione Fondo Pensione ---
with st.sidebar.expander("6. Fondo Pensione"):
    p = st.session_state.get('parametri', {})
    attiva_fondo_pensione = st.checkbox("Attiva Fondo Pensione", value=p.get('attiva_fondo_pensione', True))
    contributo_annuo_fp = st.number_input("Contributo Annuo FP (€)", min_value=0, step=100, value=p.get('contributo_annuo_fp', 3000), disabled=not attiva_fondo_pensione)
    rendimento_medio_fp = st.slider("Rendimento Medio Annuo FP (%)", 0.0, 15.0, p.get('rendimento_medio_fp', 0.04) * 100, 0.5, disabled=not attiva_fondo_pensione) / 100
    volatilita_fp = st.slider("Volatilità Annuo FP (%)", 0.0, 30.0, p.get('volatilita_fp', 0.08) * 100, 0.5, disabled=not attiva_fondo_pensione) / 100
    ter_fp = st.slider("Costo Annuo (TER) FP (%)", 0.0, 3.0, p.get('ter_fp', 0.01) * 100, 0.1, disabled=not attiva_fondo_pensione) / 100
    tassazione_rendimenti_fp = st.slider("Tassazione Rendimenti FP (%)", 0.0, 30.0, p.get('tassazione_rendimenti_fp', 0.20) * 100, 1.0, disabled=not attiva_fondo_pensione) / 100
    aliquota_finale_fp = st.slider("Aliquota Finale Ritiro FP (%)", 9.0, 23.0, p.get('aliquota_finale_fp', 0.15) * 100, 0.5, disabled=not attiva_fondo_pensione, help="La tassazione agevolata applicata al momento del ritiro del capitale o della rendita dal fondo pensione. Varia dal 15% al 9% in base agli anni di contribuzione.")
    eta_ritiro_fp = st.number_input("Età Ritiro Fondo Pensione", min_value=50, max_value=80, value=p.get('eta_ritiro_fp', 67), disabled=not attiva_fondo_pensione, help="L'età in cui maturi i requisiti per accedere al tuo fondo pensione.")
    percentuale_capitale_fp = st.slider("% Ritiro in Capitale FP", 0.0, 100.0, p.get('percentuale_capitale_fp', 0.33) * 100, 1.0, help="La parte del montante finale che desideri ritirare subito come capitale tassato. Il resto verrà convertito in una rendita mensile.", disabled=not attiva_fondo_pensione) / 100
    durata_rendita_fp_anni = st.number_input("Durata Rendita FP (Anni)", min_value=1, value=p.get('durata_rendita_fp_anni', 40), disabled=not attiva_fondo_pensione, help="Per quanti anni vuoi che venga erogata la rendita calcolata dal tuo fondo pensione.")

# --- Sezione Altre Entrate ---
with st.sidebar.expander("7. Altre Entrate"):
    p = st.session_state.get('parametri', {})
    pensione_pubblica_annua = st.number_input("Pensione Pubblica Annua (€)", min_value=0, step=500, value=p.get('pensione_pubblica_annua', 8400), help="L'importo annuo lordo della pensione statale (es. INPS) che prevedi di ricevere.")
    inizio_pensione_anni = st.number_input("Inizio Pensione (Anni da oggi)", min_value=0, value=p.get('inizio_pensione_anni', 40), help="Tra quanti anni inizierai a ricevere la pensione pubblica.")

# --- Pulsante Esecuzione ---
if st.sidebar.button("🚀 Esegui Simulazione", type="primary"):
    # Validazione allocazione
    if not np.isclose(st.session_state.portfolio["Allocazione (%)"].sum(), 100):
        st.sidebar.error("L'allocazione del portafoglio deve essere esattamente 100% per eseguire la simulazione.")
    else:
        st.session_state.parametri = {
            'eta_iniziale': eta_iniziale, 'capitale_iniziale': capitale_iniziale, 'etf_iniziale': etf_iniziale,
            'contributo_mensile_banca': contributo_mensile_banca, 'contributo_mensile_etf': contributo_mensile_etf, 
            'rendimento_medio': rendimento_medio_portfolio,
            'volatilita': volatilita_portfolio, 
            'inflazione': inflazione, 'anni_inizio_prelievo': anni_inizio_prelievo,
            'prelievo_annuo': prelievo_annuo, 'n_simulazioni': n_simulazioni, 'anni_totali': anni_totali_input,
            'strategia_prelievo': strategia_prelievo, 'percentuale_regola_4': percentuale_regola_4, 'banda_guardrail': banda_guardrail,
            'attiva_glidepath': attiva_glidepath, 'anni_glidepath': anni_glidepath, 'allocazione_finale_obbligazionario': allocazione_finale_obbligazionario,
            'tassazione_capital_gain': tassazione_capital_gain, 'attiva_fondo_pensione': attiva_fondo_pensione, 'contributo_annuo_fp': contributo_annuo_fp, 'rendimento_medio_fp': rendimento_medio_fp,
            'volatilita_fp': volatilita_fp, 'ter_fp': ter_fp, 'tassazione_rendimenti_fp': tassazione_rendimenti_fp, 'aliquota_finale_fp': aliquota_finale_fp,
            'eta_ritiro_fp': eta_ritiro_fp, 'percentuale_capitale_fp': percentuale_capitale_fp, 'durata_rendita_fp_anni': durata_rendita_fp_anni,
            'pensione_pubblica_annua': pensione_pubblica_annua, 'inizio_pensione_anni': inizio_pensione_anni
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

    # --- Riepilogo Statistico ---
    st.header("Riepilogo Statistico Chiave")

    # Calcoli per le nuove metriche
    patrimonio_iniziale_totale = params['capitale_iniziale'] + params['etf_iniziale']
    contributi_versati = stats['contributi_totali_versati_mediano_nominale']
    patrimonio_finale_nominale = stats['patrimonio_finale_mediano_nominale']
    guadagni_da_investimento = patrimonio_finale_nominale - contributi_versati - patrimonio_iniziale_totale
    
    reddito_annuo_reale_pensione = st.session_state.risultati['statistiche_prelievi']['totale_reale_medio_annuo']
    anni_di_spesa_coperti = (stats['patrimonio_finale_mediano_reale'] / reddito_annuo_reale_pensione) if reddito_annuo_reale_pensione > 0 else float('inf')


    st.write("---")
    st.markdown("##### Il Tuo Percorso Finanziario in Numeri")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Patrimonio Iniziale", f"€ {patrimonio_iniziale_totale:,.0f}",
        help="La somma del capitale che hai all'inizio della simulazione."
    )
    col2.metric(
        "Contributi Totali Versati", f"€ {contributi_versati:,.0f}",
        help="La stima di tutto il denaro che verserai di tasca tua durante la fase di accumulo. Questo è il tuo sacrificio."
    )
    col3.metric(
        "Guadagni da Investimento", f"€ {guadagni_da_investimento:,.0f}",
        delta=f"{((guadagni_da_investimento / contributi_versati) * 100) if contributi_versati > 0 else 0:,.0f}% vs Contributi",
        help="La ricchezza generata dal solo effetto dei rendimenti di mercato (interesse composto). Questa è la ricompensa per il rischio e la pazienza."
    )
    col4.metric(
        "Patrimonio Finale in Anni di Spesa", f"{anni_di_spesa_coperti:,.1f} Anni",
        help=f"Il tuo patrimonio finale reale mediano, tradotto in quanti anni del tuo tenore di vita pensionistico (€{reddito_annuo_reale_pensione:,.0f}/anno) può coprire."
    )

    st.write("---")
    st.markdown("##### Risultati Finali della Simulazione (Patrimonio Nominale)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Patrimonio Finale Mediano (50°)", f"€ {stats['patrimonio_finale_mediano_nominale']:,.0f}",
        help="Il risultato che si trova esattamente nel mezzo di tutti gli scenari. È la stima più realistica."
    )
    col2.metric(
        "Patrimonio Finale (Top 10% - 90°)", f"€ {stats['patrimonio_finale_top_10_nominale']:,.0f}",
        help="Lo scenario 'da sogno'. C'è solo un 10% di probabilità che le cose vadano meglio di così."
    )
    col3.metric(
        "Patrimonio Finale (Peggior 10% - 10°)", f"€ {stats['patrimonio_finale_peggior_10_nominale']:,.0f}",
        help="Lo scenario 'notte insonne'. C'è un 10% di probabilità che le cose vadano peggio di così."
    )
    col4.metric(
        "Patrimonio Reale Finale Mediano (50°)", f"€ {stats['patrimonio_finale_mediano_reale']:,.0f}",
        help="Il potere d'acquisto mediano del tuo patrimonio a fine piano, espresso in Euro di oggi. La metrica più importante."
    )


    st.write("---")
    st.markdown("##### Indicatori di Rischio del Piano")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Probabilità di Fallimento", f"{stats['probabilita_fallimento']:.2%}",
        delta=f"{-stats['probabilita_fallimento']:.2%}", delta_color="inverse",
        help="La probabilità di finire i soldi prima della fine della simulazione."
    )
    col2.metric(
        "Drawdown Massimo Peggiore", f"{stats['drawdown_massimo_peggiore']:.2%}",
        delta=f"{stats['drawdown_massimo_peggiore']:.2%}", delta_color="inverse",
        help="La perdita percentuale più grande dal picco, nello scenario peggiore. Misura il 'dolore' massimo che potresti sopportare."
    )
    col3.metric(
        "Sharpe Ratio Medio", f"{stats['sharpe_ratio_medio']:.2f}",
        help="Il rendimento ottenuto per ogni unità di rischio. Un valore più alto è meglio (sopra 1 è ottimo)."
    )


    # --- Riepilogo Entrate in Pensione ---
    st.write("---")
    st.header("Riepilogo Entrate in Pensione (Valori Reali Medi)")
    st.markdown("Queste metriche mostrano il tenore di vita **medio annuo** che puoi aspettarti durante la fase di ritiro, espresso nel potere d'acquisto di oggi.")
    stats_prelievi = st.session_state.risultati['statistiche_prelievi']

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

    # --- Sezione di Spiegazione e Grafico Composizione ---
    with st.expander("🔍 Guida alla Lettura: Perché il mio piano ha successo (o fallisce)?"):
        st.markdown("""
        **Perché la probabilità di fallimento è spesso 0%?**
        Se vedi un fallimento dello 0%, non significa che il simulatore sia rotto. Significa che, date le ipotesi che hai inserito (contributi costanti, orizzonte lungo, rendimenti positivi), il tuo piano è matematicamente molto solido. Questo grafico ti aiuta a capire perché.
        """)
        
        st.plotly_chart(plot_wealth_composition_chart(patrimonio_iniziale_totale, contributi_versati, guadagni_da_investimento), use_container_width=True)

        st.markdown("""
        Come puoi vedere, nel lungo periodo, i **Guadagni da Investimento** (la ricompensa per il rischio e la pazienza) spesso superano persino il totale dei contributi che hai versato. Questo è l'effetto dell'**interesse composto**: i tuoi guadagni iniziano a generare altri guadagni, in un circolo virtuoso che accelera la crescita del tuo patrimonio.

        **Limiti del Modello da Tenere a Mente:**
        Questo simulatore è un potente strumento matematico, ma non può prevedere il futuro o la vita reale al 100%. Ricorda che:
        - **Non considera shock improvvisi:** La perdita del lavoro, una spesa medica imprevista, o l'impossibilità di contribuire per alcuni anni non sono modellizzati.
        - **Non considera l'emotività:** Non tiene conto del rischio di vendere in preda al panico durante un crollo di mercato.
        - **I rendimenti sono un'ipotesi:** I rendimenti e la volatilità che hai inserito sono stime a lungo termine. Il futuro potrebbe essere diverso.
        - **Le tasse sono semplificate:** Il modello usa un'aliquota fissa del 26% sul capital gain, senza considerare scaglioni, minusvalenze pregresse o altre ottimizzazioni fiscali complesse.

        Usa questo strumento come una mappa per definire la direzione, non come un GPS che prevede la destinazione al centimetro.
        """)


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
        eta_pensione_pubblica = params['eta_iniziale'] + params['inizio_pensione_anni']
        eta_ritiro_fp = params['eta_ritiro_fp']

        st.subheader(f"Dalla pensione (a {eta_pensionamento} anni) in poi")
        
        testo_decumulo = f"""
        A partire da **{eta_pensionamento} anni**, smetti di versare e inizi a **prelevare dal tuo patrimonio** per sostenere il tuo tenore di vita. 
        A questo si aggiungeranno le altre fonti di reddito che hai configurato:
        - La **pensione pubblica** a partire da **{eta_pensione_pubblica} anni**.
        """
        if params['attiva_fondo_pensione']:
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
            'Liquidazione Capitale FP (Nom.)', 'Prelievo Effettivo (Reale)', 'Pensione Pubblica (Reale)', 
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
                "Liquidazione Capitale FP (Nom.)": st.column_config.NumberColumn(format="€ %.0f"),
                "Prelievo Effettivo (Reale)": st.column_config.NumberColumn(format="€ %.0f"),
                "Pensione Pubblica (Reale)": st.column_config.NumberColumn(format="€ %.0f"),
                "Rendita FP (Reale)": st.column_config.NumberColumn(format="€ %.0f"),
                "Entrate Anno (Reali)": st.column_config.NumberColumn(format="€ %.0f"),
                "Saldo Conto Fine Anno (Reale)": st.column_config.NumberColumn(format="€ %.0f"),
                "Valore ETF Fine Anno (Reale)": st.column_config.NumberColumn(format="€ %.0f")
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