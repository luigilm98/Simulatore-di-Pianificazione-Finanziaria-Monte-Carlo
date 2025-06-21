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
with st.sidebar.expander("Liquidità e Portafoglio Iniziale", expanded=True):
    st.session_state['patrimonio_iniziale'] = st.number_input(
        'Liquidità sul Conto Corrente (€)', 
        min_value=0.0, 
        value=st.session_state.get('patrimonio_iniziale', 10000.0), 
        step=1000.0,
        help="Inserisci la somma di denaro che hai attualmente disponibile sul conto corrente e che vuoi includere nella simulazione. Questo rappresenta il tuo punto di partenza finanziario."
    )
    st.session_state['valore_attuale_investimenti'] = st.number_input(
        'Valore Attuale del Portafoglio ETF (€)', 
        min_value=0.0, 
        value=st.session_state.get('valore_attuale_investimenti', 5000.0), 
        step=1000.0,
        help="Inserisci il valore totale di mercato del tuo portafoglio di ETF (o altri investimenti simili). Questo capitale, sommato alla liquidità, costituisce il tuo patrimonio iniziale."
    )

# --- Sezione Portafoglio ETF ---
with st.sidebar.expander("Costruttore di Portafoglio ETF", expanded=True):
    st.markdown("Modifica l'allocazione, il TER e le stime di rendimento/volatilità per ogni ETF.")
    
    edited_portfolio = st.data_editor(
        st.session_state.portfolio,
        column_config={
            "Allocazione (%)": st.column_config.NumberColumn(format="%.2f%%", min_value=0, max_value=100),
            "TER (%)": st.column_config.NumberColumn(format="%.2f%%", min_value=0),
            "Rendimento Atteso (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "Volatilità Attesa (%)": st.column_config.NumberColumn(format="%.2f%%"),
        },
        num_rows="dynamic",
        key="portfolio_editor"
    )

    total_allocation = edited_portfolio["Allocazione (%)"].sum()
    if not np.isclose(total_allocation, 100):
        st.warning(f"Allocazione totale: {total_allocation}%. Aggiusta per raggiungere il 100%.", disabled=edited_portfolio.columns.drop('Allocazione (%)'))
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

# --- Sezione Strategie di Prelievo ---
with st.sidebar.expander("Gestione della Fase di Decumulo (Pensione)", expanded=False):
    st.session_state['anni_prelievo'] = st.number_input(
        'Anni all\'Inizio dei Prelievi (Età)',
        min_value=1,
        max_value=100,
        value=st.session_state.get('anni_prelievo', 35),
        step=1,
        help="Indica l'età in cui prevedi di smettere di lavorare e iniziare a prelevare dal tuo patrimonio per vivere. Questo segna l'inizio della fase di 'decumulo'."
    )
    st.session_state['percentuale_prelievo_iniziale'] = st.slider(
        'Percentuale di Prelievo Annuale Iniziale (%)',
        min_value=0.0,
        max_value=10.0,
        value=st.session_state.get('percentuale_prelievo_iniziale', 4.0),
        step=0.1,
        help="Quale percentuale del tuo patrimonio totale prevedi di prelevare il primo anno di pensione? La 'regola del 4%' è un punto di partenza comune, ma puoi aggiustarla. Ogni anno successivo, l'importo prelevato verrà adeguato all'inflazione."
    )
    st.session_state['usa_safe_withdrawal_rate'] = st.checkbox(
        "Usa Tasso di Prelievo Massimo Sostenibile (SWR)",
        value=st.session_state.get('usa_safe_withdrawal_rate', True),
        help="Se spuntato, il simulatore calcolerà per te il tasso di prelievo massimo che puoi sostenere con una probabilità di successo del 95%. Questo ignorerà il valore da te impostato sopra, dandoti un'indicazione più scientifica."
    )

# --- Sezione Asset Allocation Dinamica (Glidepath) ---
with st.sidebar.expander("4. Asset Allocation Dinamica (Glidepath)"):
    p = st.session_state.get('parametri', {})
    attiva_glidepath = st.checkbox("Attiva Glidepath", value=p.get('attiva_glidepath', True), help="Se attivato, il simulatore ridurrà progressivamente l'esposizione azionaria (ETF) a favore della liquidità con l'avvicinarsi e durante la pensione, per ridurre il rischio.")
    inizio_glidepath_anni = st.number_input("Inizio Glidepath (Anni da oggi)", min_value=0, value=p.get('inizio_glidepath_anni', 20), disabled=not attiva_glidepath, help="L'anno in cui inizi a rendere il tuo portafoglio più conservativo. Spesso si imposta 10-15 anni prima della pensione.")
    fine_glidepath_anni = st.number_input("Fine Glidepath (Anni da oggi)", min_value=0, value=p.get('fine_glidepath_anni', 40), disabled=not attiva_glidepath, help="L'anno in cui raggiungi l'allocazione finale desiderata. Solitamente coincide con l'inizio della pensione o pochi anni dopo.")
    allocazione_etf_finale = st.slider(
        "Allocazione ETF Finale (%)", 0.0, 100.0, p.get('allocazione_etf_finale', 0.333) * 100, 1.0,
        help="La percentuale di patrimonio che rimarrà investita in ETF alla fine del percorso di de-risking. Il resto sarà liquidità. Un valore comune è tra il 30% e il 50%.",
        disabled=not attiva_glidepath
    ) / 100

# --- Sezione Tassazione e Costi ---
with st.sidebar.expander("5. Tassazione e Costi (Italia)"):
    p = st.session_state.get('parametri', {})
    tassazione_capital_gain = st.slider("Tassazione Capital Gain (%)", 0.0, 50.0, p.get('tassazione_capital_gain', 0.26) * 100, 1.0, help="L'aliquota applicata ai profitti derivanti dalla vendita di ETF. In Italia è tipicamente il 26%.") / 100
    imposta_bollo_titoli = st.slider("Imposta di Bollo Titoli (annua, %)", 0.0, 1.0, p.get('imposta_bollo_titoli', 0.002) * 100, 0.01, help="Tassa patrimoniale annuale sul valore totale del tuo portafoglio titoli. In Italia è lo 0,2%.") / 100
    imposta_bollo_conto = st.number_input("Imposta di Bollo Conto (>5k€)", min_value=0, value=p.get('imposta_bollo_conto', 34), help="Imposta fissa annuale sui conti correnti con giacenza media superiore a 5.000€. In Italia è 34,20€.")
    # ter_etf è ora calcolato dal portafoglio
    costo_fisso_etf_mensile = st.number_input("Costo Fisso Deposito Titoli (€/mese)", min_value=0.0, value=p.get('costo_fisso_etf_mensile', 4.0), step=0.5, help="Eventuali costi fissi mensili o annuali addebitati dal tuo broker per il mantenimento del conto titoli.")

# --- Sezione Fondo Pensione ---
with st.sidebar.expander("Opzioni Fondo Pensione Complementare", expanded=False):
    st.session_state['attiva_fondo_pensione'] = st.checkbox(
        "Includi un Fondo Pensione nel Piano",
        value=st.session_state.get('attiva_fondo_pensione', False),
        help="Spunta questa casella se versi contributi in un fondo pensione complementare e vuoi includerlo nella simulazione."
    )
    if st.session_state.get('attiva_fondo_pensione', False):
        st.session_state['valore_attuale_fp'] = st.number_input(
            'Valore Attuale del Fondo Pensione (€)',
            min_value=0.0,
            value=st.session_state.get('valore_attuale_fp', 0.0),
            step=500.0,
            help="Inserisci il montante già accumulato nel tuo fondo pensione complementare."
        )
        st.session_state['contributo_annuo_fp'] = st.number_input(
            'Contributo Annuo al Fondo Pensione (€)',
            min_value=0.0,
            value=st.session_state.get('contributo_annuo_fp', 1500.0),
            step=100.0,
            help="L'importo totale che versi ogni anno nel fondo pensione (include i tuoi versamenti, quelli del datore di lavoro e il TFR)."
        )
        st.session_state['rendimento_medio_fp'] = st.slider(
            'Rendimento Medio Annuo Stimato del Fondo Pensione (%)',
            min_value=0.0,
            max_value=10.0,
            value=st.session_state.get('rendimento_medio_fp', 4.0),
            step=0.5,
            help="Il rendimento medio annuo che ti aspetti dal tuo comparto del fondo pensione, al netto dei costi di gestione. Controlla il KID o la scheda del tuo fondo per una stima realistica."
        )
        st.session_state['eta_ritiro_fp'] = st.number_input(
            'Età di Pensionamento (Accesso al Fondo Pensione)',
            min_value=50,
            max_value=75,
            value=st.session_state.get('eta_ritiro_fp', 67),
            step=1,
            help="L'età in cui maturerai i requisiti per accedere alla prestazione del fondo pensione (solitamente coincide con l'età della pensione di vecchiaia)."
        )
        st.session_state['percentuale_liquidazione_fp'] = st.slider(
            'Percentuale del Fondo Liquidato in Capitale (%)',
            min_value=0,
            max_value=50,
            value=st.session_state.get('percentuale_liquidazione_fp', 50),
            step=5,
            help="Per legge, puoi richiedere fino al 50% del montante accumulato come capitale immediato al pensionamento. Il resto verrà convertito in una rendita annuale. Imposta qui la percentuale desiderata."
        )

# --- Sezione Altre Entrate ---
with st.sidebar.expander("Opzioni Pensione Pubblica (INPS)", expanded=False):
    st.session_state['attiva_pensione_pubblica'] = st.checkbox(
        "Includi la Pensione Pubblica nel Piano",
        value=st.session_state.get('attiva_pensione_pubblica', False),
        help="Spunta questa casella se vuoi includere una stima della tua pensione pubblica futura nei calcoli."
    )
    if st.session_state.get('attiva_pensione_pubblica', False):
        st.session_state['eta_pensione_pubblica'] = st.number_input(
            'Età di Accesso alla Pensione Pubblica',
            min_value=60,
            max_value=75,
            value=st.session_state.get('eta_pensione_pubblica', 67),
            step=1,
            help="L'età in cui prevedi di ricevere il tuo primo assegno pensionistico dall'INPS."
        )
        st.session_state['importo_pensione_pubblica_annua'] = st.number_input(
            'Importo Annuo Stimato della Pensione Pubblica (€)',
            min_value=0.0,
            value=st.session_state.get('importo_pensione_pubblica_annua', 12000.0),
            step=500.0,
            help="Inserisci una stima dell'importo annuo lordo della tua pensione pubblica. Puoi usare il simulatore 'La Mia Pensione' sul sito dell'INPS per ottenere una stima personalizzata."
        )

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
            'attiva_glidepath': attiva_glidepath, 'inizio_glidepath_anni': inizio_glidepath_anni, 'fine_glidepath_anni': fine_glidepath_anni,
            'allocazione_etf_finale': allocazione_etf_finale,
            'tassazione_capital_gain': tassazione_capital_gain, 'imposta_bollo_titoli': imposta_bollo_titoli, 'imposta_bollo_conto': imposta_bollo_conto,
            'ter_etf': ter_etf_portfolio, 
            'costo_fisso_etf_mensile': costo_fisso_etf_mensile,
            'attiva_fondo_pensione': attiva_fondo_pensione, 'contributo_annuo_fp': contributo_annuo_fp, 'rendimento_medio_fp': rendimento_medio_fp,
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
    contributi_versati = stats['totale_contributi_versati_nominale_medio']
    guadagni_da_investimento = stats['patrimonio_nominale_medio_finale'] - patrimonio_iniziale_totale - contributi_versati
    patrimonio_finale_in_anni_di_spesa = stats['patrimonio_reale_medio_finale'] / params['prelievo_target_reale_iniziale'] if params['prelievo_target_reale_iniziale'] > 0 else 0

    st.markdown("---")
    st.markdown("<h2 style='text-align: center; font-size: 28px;'>Il Tuo Percorso Finanziario in Numeri</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #a0a0a0;'>Una sintesi dei risultati chiave della tua simulazione. Tutti i valori monetari sono medie di tutti gli scenari possibili.</p>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Patrimonio Iniziale Totale",
            value=f"€ {patrimonio_iniziale_totale:,.0f}",
            help="La somma della tua liquidità iniziale e del valore del portafoglio ETF. È il tuo punto di partenza."
        )
    with col2:
        st.metric(
            label="Contributi Totali Versati",
            value=f"€ {contributi_versati:,.0f}",
            help="La somma totale di tutti i contributi mensili che hai versato nel portafoglio (e nel fondo pensione, se attivo) lungo tutta la fase di accumulo. Questo è il capitale che hai messo di tasca tua."
        )
    with col3:
        st.metric(
            label="Guadagni da Investimento (Capital Gain)",
            value=f"€ {guadagni_da_investimento:,.0f}",
            delta=f"{((guadagni_da_investimento / contributi_versati) * 100):.0f}% vs Contributi" if contributi_versati > 0 else "N/A",
            help="Il profitto generato dai tuoi investimenti (interessi, dividendi, plusvalenze), al netto dei costi e delle tasse. È il rendimento del tuo capitale."
        )
    with col4:
        st.metric(
            label="Patrimonio Finale in Anni di Spesa",
            value=f"{patrimonio_finale_in_anni_di_spesa:.1f} Anni",
            help="Una misura intuitiva della tua ricchezza: per quanti anni potresti vivere prelevando un importo pari al tuo obiettivo di spesa iniziale, usando solo il patrimonio reale medio finale. Più alto è, meglio è."
        )

    st.markdown("---")
    
    st.markdown("<h3 style='text-align: left; font-size: 22px;'>Risultati Finali della Simulazione (Patrimonio Nominale)</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: left; color: #a0a0a0;'>Come potrebbe apparire il tuo patrimonio alla fine dell'orizzonte temporale, senza contare l'inflazione. Utile per confronti, ma il valore reale è più importante.</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="Patrimonio Finale Medio (50° percentile)",
            value=f"€ {stats['patrimonio_nominale_medio_finale']:,.0f}",
            help="Il risultato mediano. C'è il 50% di probabilità che il tuo patrimonio finale sia superiore a questa cifra e il 50% che sia inferiore."
        )
    with col2:
        st.metric(
            label="Scenario Ottimistico (90° percentile)",
            value=f"€ {stats['patrimonio_nominale_percentile_90']:,.0f}",
            help="Uno scenario fortunato. C'è solo il 10% di probabilità che il tuo patrimonio superi questa cifra."
        )
    with col3:
        st.metric(
            label="Scenario Pessimistico (10° percentile)",
            value=f"€ {stats['patrimonio_nominale_percentile_10']:,.0f}",
            help="Uno scenario sfortunato. C'è il 10% di probabilità che il tuo patrimonio sia inferiore a questa cifra. È un buon valore per valutare il rischio del tuo piano."
        )

    st.markdown("<h3 style='text-align: left; font-size: 22px;'>Risultati Finali della Simulazione (Patrimonio Reale)</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: left; color: #a0a0a0;'>Questo è il valore che conta davvero: il potere d'acquisto del tuo patrimonio alla fine della simulazione, tenendo conto dell'erosione dell'inflazione.</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="Patrimonio Reale Finale Medio (50° percentile)",
            value=f"€ {stats['patrimonio_reale_medio_finale']:,.0f}",
            help="Il potere d'acquisto mediano del tuo patrimonio alla fine del piano. Questo è l'indicatore più importante del tuo successo finanziario."
        )
    with col2:
        st.metric(
            label="Patrimonio Reale Finale Ottimistico (90° percentile)",
            value=f"€ {stats['patrimonio_reale_percentile_90']:,.0f}",
            help="Il potere d'acquisto del tuo patrimonio in uno scenario fortunato (10% di probabilità di superarlo)."
        )
    with col3:
        st.metric(
            label="Patrimonio Reale Finale Pessimistico (10° percentile)",
            value=f"€ {stats['patrimonio_reale_percentile_10']:,.0f}",
            help="Il potere d'acquisto del tuo patrimonio in uno scenario sfortunato (10% di probabilità di essere inferiore). Se questo valore è sufficiente a coprire le tue necessità, il tuo piano è molto robusto."
        )

    st.markdown("---")
    st.markdown("<h3 style='text-align: left; font-size: 22px;'>Indicatori di Rischio e Sostenibilità del Piano</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: left; color: #a0a0a0;'>Queste metriche ti aiutano a capire la robustezza del tuo piano e le probabilità di raggiungere i tuoi obiettivi.</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="Probabilità di Successo del Piano",
            value=f"{stats['probabilita_successo'] * 100:.0f}%",
            delta=f"{stats['probabilita_successo'] * 100 - 100:.0f}% vs Obiettivo",
            help="La percentuale di simulazioni in cui il tuo patrimonio non si è esaurito prima della fine dell'orizzonte temporale. Un valore > 95% è considerato molto robusto."
        )
    with col2:
        st.metric(
            label="Drawdown Massimo Peggiore (Scenario Reale)",
            value=f"{stats['max_drawdown_reale_peggiore'] * 100:.0f}%",
            help="Il più grande calo percentuale del tuo portafoglio (dal picco al minimo) avvenuto in una singola simulazione, considerando i valori reali. Ti dà un'idea della massima perdita che avresti potuto sopportare psicologicamente."
        )
    with col3:
        sharpe_ratio_medio_ann = stats.get('sharpe_ratio_medio_annualizzato', 0.0)
        st.metric(
            label="Sharpe Ratio Medio (Annualizzato)",
            value=f"{sharpe_ratio_medio_ann:.2f}",
            help="Misura il rendimento del tuo portafoglio in rapporto al rischio che ti sei assunto. Un valore più alto indica una migliore performance corretta per il rischio. (Valori > 1 sono considerati ottimi)."
        )

    st.markdown("---")
    st.markdown(f"""
    Basandosi sulla tua età iniziale ({params['eta_iniziale']} anni) e sull'inizio dei prelievi tra {params['anni_prelievo']} anni, la tua **fase di accumulo** durerà fino a quando avrai **{int(params['eta_iniziale'] + params['anni_prelievo'])} anni**.
    Durante questo periodo, il tuo obiettivo sarà far crescere il patrimonio.

    Successivamente, inizierà la **fase di decumulo (o pensione)**. Ecco un riepilogo delle tue entrate medie annuali durante questa fase, espresse nel potere d'acquisto di oggi (valori reali):
    """)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Prelievo Medio dal Patrimonio", 
            value=f"€ {stats.get('prelievo_reale_medio_decumulo', 0):,.0f}",
            help="L'importo medio annuo che preleverai dal tuo portafoglio di investimenti durante la pensione, espresso in potere d'acquisto di oggi."
        )
    with col2:
        st.metric(
            label="Pensione Pubblica Annua", 
            value=f"€ {stats.get('pensione_pubblica_reale_media_decumulo', 0):,.0f}",
            help="La tua pensione pubblica media annua, al netto dell'inflazione."
        )
    with col3:
        st.metric(
            label="Rendita Media da Fondo Pensione", 
            value=f"€ {stats.get('rendita_fp_reale_media_decumulo', 0):,.0f}",
            help="La rendita media annua che riceverai dal tuo fondo pensione complementare, al netto dell'inflazione."
        )
    with col4:
        st.metric(
            label="TOTALE ENTRATE MEDIE ANNUE", 
            value=f"€ {stats.get('totale_entrate_reali_medie_decumulo', 0):,.0f}",
            help="La somma di tutte le tue entrate medie annue in pensione. Questa cifra rappresenta il tuo tenore di vita medio annuo reale."
        )
    
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