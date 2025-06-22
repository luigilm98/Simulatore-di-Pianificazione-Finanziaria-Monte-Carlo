import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime
import plotly.express as px

import simulation_engine as engine

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(
    page_title="Simulatore Finanziario Monte Carlo",
    page_icon="✈️",
    layout="wide"
)

# --- FUNZIONI HELPER ---

class NpEncoder(json.JSONEncoder):
    """ Encoder JSON custom per gestire i tipi di dati NumPy. """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def save_simulation(name, params, results):
    """ Salva i risultati completi di una simulazione in un file JSON. """
    # Assicura che la directory esista prima di salvare
    history_dir = 'simulation_history'
    os.makedirs(history_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{name.replace(' ', '_')}.json"
    filepath = os.path.join(history_dir, filename)
    
    data_to_save = {
        "simulation_name": name,
        "timestamp": timestamp,
        "parameters": params,
        "results": results
    }
    
    with open(filepath, 'w') as f:
        json.dump(data_to_save, f, cls=NpEncoder, indent=4)
    st.success(f"Risultati salvati con successo in `{filepath}`")

def load_simulation_files():
    """ Carica la lista dei file di simulazione salvati. """
    history_dir = 'simulation_history'
    if not os.path.exists(history_dir):
        return []
    files = [f for f in os.listdir(history_dir) if f.endswith('.json')]
    return sorted(files, reverse=True)

def load_simulation_data(filename):
    """ Carica i dati da un file di simulazione JSON. """
    filepath = os.path.join('simulation_history', filename)
    with open(filepath, 'r') as f:
        return json.load(f)

# --- FUNZIONI DI PLOTTING ---

def plot_wealth_composition_chart(initial, contributions, gains):
    """Crea un grafico a barre per mostrare la composizione della ricchezza."""
    labels = ['Patrimonio Iniziale', 'Contributi Versati', 'Guadagni da Investimento']
    values = [initial, contributions, gains]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    fig = go.Figure(data=[go.Bar(
        x=labels, 
        y=values,
        marker_color=colors,
        text=[f"€{v:,.0f}" for v in values],
        textposition='auto',
        hovertemplate='Fonte: %{x}<br>Valore: €%{y:,.0f}<extra></extra>'
    )])
    
    fig.update_layout(
        title_text='Da Dove Viene la Tua Ricchezza? (Scenario Mediano)',
        yaxis_title_text='Euro (€)',
        xaxis_title_text='Fonte del Patrimonio',
        bargap=0.4,
        yaxis_tickformat="€,d"
    )
    return fig

def plot_wealth_summary_chart(data, title, y_title, anni_totali, eta_iniziale, anni_inizio_prelievo, color_median='#C00000', color_fill='#C00000'):
    """
    Crea il grafico principale che mostra l'evoluzione del patrimonio
    con gli intervalli di confidenza (percentili).
    """
    fig = go.Figure()
    
    # L'asse x (mesi) deve avere la stessa lunghezza dei dati
    mesi = np.arange(data.shape[1])
    x_axis_labels = eta_iniziale + mesi / 12

    p10 = np.percentile(data, 10, axis=0)
    p25 = np.percentile(data, 25, axis=0)
    p50 = np.median(data, axis=0)
    p75 = np.percentile(data, 75, axis=0)
    p90 = np.percentile(data, 90, axis=0)

    # Funzione helper per convertire hex in rgba per il fill
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    rgb_fill = hex_to_rgb(color_fill)

    # Aree di confidenza
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_axis_labels, x_axis_labels[::-1]]),
        y=np.concatenate([p90, p10[::-1]]),
        fill='toself',
        fillcolor=f'rgba({rgb_fill[0]}, {rgb_fill[1]}, {rgb_fill[2]}, 0.2)',
        line={'color': 'rgba(255,255,255,0)'},
        name='10-90 Percentile',
        hoverinfo='none'
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_axis_labels, x_axis_labels[::-1]]),
        y=np.concatenate([p75, p25[::-1]]),
        fill='toself',
        fillcolor=f'rgba({rgb_fill[0]}, {rgb_fill[1]}, {rgb_fill[2]}, 0.4)',
        line={'color': 'rgba(255,255,255,0)'},
        name='25-75 Percentile',
        hoverinfo='none'
    ))

    # Linea mediana
    fig.add_trace(go.Scatter(
        x=x_axis_labels, y=p50, mode='lines',
        name='Scenario Mediano (50°)',
        line={'width': 3, 'color': color_median},
        hovertemplate='Età %{x:.1f}<br>Patrimonio Mediano: €%{y:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Età",
        yaxis_title=y_title,
        yaxis_tickformat="€,d",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    return fig

def plot_spaghetti_chart(data, title, y_title, anni_totali, eta_iniziale, anni_inizio_prelievo, color_median='#C00000'):
    """Crea un grafico 'spaghetti' con le singole simulazioni e la mediana."""
    fig = go.Figure()
    anni_asse_x = eta_iniziale + np.linspace(0, anni_totali, data.shape[1])

    # Mostra un sottoinsieme di simulazioni per non appesantire il grafico
    n_sim_da_mostrare = min(50, data.shape[0])
    indici_da_mostrare = np.random.choice(data.shape[0], size=n_sim_da_mostrare, replace=False)

    # Usa una palette di colori per le linee
    color_palette = px.colors.qualitative.Plotly

    for i, idx in enumerate(indici_da_mostrare):
        fig.add_trace(go.Scatter(
            x=anni_asse_x, y=data[idx, :], mode='lines',
            line={'width': 1.5, 'color': color_palette[i % len(color_palette)]},
            opacity=0.6,
            hoverinfo='none',
            showlegend=False,
            name=f'Simulazione {i}'
        ))

    # Aggiungi la mediana in evidenza
    median_data = np.median(data, axis=0)
    fig.add_trace(go.Scatter(
        x=anni_asse_x, y=median_data, mode='lines',
        name='Scenario Mediano (50°)',
        line={'width': 4, 'color': color_median},
        hovertemplate='Età %{x:.1f}<br>Patrimonio Mediano: €%{y:,.0f}<extra></extra>'
    ))
            
    fig.update_layout(
        title=title,
        xaxis_title="Età",
        yaxis_title=y_title,
        yaxis_tickformat="€,d",
        hovermode="x unified",
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    fig.add_vline(x=eta_iniziale + anni_inizio_prelievo, line_width=2, line_dash="dash", line_color="grey", annotation_text="Inizio Prelievi")
    return fig

def plot_income_cone_chart(data, anni_totali, anni_inizio_prelievo, eta_iniziale):
    """Crea un grafico a cono per mostrare l'evoluzione del reddito reale annuo."""
    fig = go.Figure()
    
    # Calcola i percentili per il reddito
    p10 = np.percentile(data, 10, axis=0)
    p25 = np.percentile(data, 25, axis=0)
    p50 = np.median(data, axis=0)
    p75 = np.percentile(data, 75, axis=0)
    p90 = np.percentile(data, 90, axis=0)
    
    # Asse x (età)
    eta_asse_x = eta_iniziale + np.arange(data.shape[1])
    
    # Aree di confidenza
    fig.add_trace(go.Scatter(
        x=np.concatenate([eta_asse_x, eta_asse_x[::-1]]),
        y=np.concatenate([p90, p10[::-1]]),
        fill='toself',
        fillcolor='rgba(0, 123, 255, 0.2)',
        line={'color': 'rgba(255,255,255,0)'},
        name='10-90 Percentile',
        hoverinfo='none'
    ))
    
    fig.add_trace(go.Scatter(
        x=np.concatenate([eta_asse_x, eta_asse_x[::-1]]),
        y=np.concatenate([p75, p25[::-1]]),
        fill='toself',
        fillcolor='rgba(0, 123, 255, 0.4)',
        line={'color': 'rgba(255,255,255,0)'},
        name='25-75 Percentile',
        hoverinfo='none'
    ))
    
    # Linea mediana
    fig.add_trace(go.Scatter(
        x=eta_asse_x, y=p50, mode='lines',
        name='Reddito Mediano',
        line={'width': 3, 'color': '#007bff'},
        hovertemplate='Età %{x}<br>Reddito Annuo: €%{y:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Evoluzione del Reddito Annuo Reale in Pensione',
        xaxis_title="Età",
        yaxis_title="Reddito Annuo Reale (€)",
        yaxis_tickformat="€,d",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    # Aggiungi linea verticale per l'inizio dei prelievi
    fig.add_vline(x=eta_iniziale + anni_inizio_prelievo, line_width=2, line_dash="dash", line_color="grey", annotation_text="Inizio Prelievi")
    
    return fig

def plot_worst_scenarios_chart(patrimoni_finali, data, anni_totali, eta_iniziale):
    """Crea un grafico che mostra i 10% scenari peggiori."""
    fig = go.Figure()
    
    # Trova il 10% degli scenari peggiori
    n_worst = max(1, int(data.shape[0] * 0.1))
    worst_indices = np.argsort(patrimoni_finali)[:n_worst]
    worst_data = data[worst_indices, :]
    
    anni_asse_x = eta_iniziale + np.linspace(0, anni_totali, data.shape[1])
    
    # Disegna le linee degli scenari peggiori
    for i in range(worst_data.shape[0]):
        fig.add_trace(go.Scatter(
            x=anni_asse_x, y=worst_data[i, :], mode='lines',
            line={'width': 1, 'color': 'rgba(220, 53, 69, 0.5)'}, # Rosso con trasparenza
            hoverinfo='none',
            showlegend=False,
            name=f'Scenario Peggiore {i}'
        ))
    
    # Aggiungi la mediana DEI SOLI SCENARI PEGGIORI per riferimento
    median_worst_data = np.median(worst_data, axis=0)
    fig.add_trace(go.Scatter(
        x=anni_asse_x, y=median_worst_data, mode='lines',
        name='Mediana Scenari Peggiori',
        line={'width': 3, 'color': '#dc3545'}, # Rosso più scuro per la mediana
        hovertemplate='Età %{x:.1f}<br>Patrimonio Mediano (Peggiori): €%{y:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Analisi degli Scenari Peggiori (10% più sfortunati)',
        xaxis_title="Età",
        yaxis_title="Patrimonio Reale (€)",
        yaxis_tickformat="€,d",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig

def plot_wealth_composition_over_time_nominal(dati_tabella, anni_totali, eta_iniziale):
    """Crea un grafico che mostra la composizione del patrimonio nel tempo (valori nominali)."""
    fig = go.Figure()
    
    anni_asse_x = eta_iniziale + np.arange(anni_totali + 1)
    
    # Estrai i dati di composizione
    saldo_banca = dati_tabella.get('saldo_banca_nominale', np.zeros(anni_totali + 1))
    saldo_etf = dati_tabella.get('saldo_etf_nominale', np.zeros(anni_totali + 1))
    saldo_fp = dati_tabella.get('saldo_fp_nominale', np.zeros(anni_totali + 1))
    
    # Crea il grafico a area stack
    fig.add_trace(go.Scatter(
        x=anni_asse_x, y=saldo_banca, mode='lines',
        fill='tonexty', name='Liquidità',
        line={'color': '#28a745'},
        hovertemplate='Età %{x}<br>Liquidità: €%{y:,.0f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=anni_asse_x, y=saldo_etf, mode='lines',
        fill='tonexty', name='ETF',
        line={'color': '#007bff'},
        hovertemplate='Età %{x}<br>ETF: €%{y:,.0f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=anni_asse_x, y=saldo_fp, mode='lines',
        fill='tonexty', name='Fondo Pensione',
        line={'color': '#ffc107'},
        hovertemplate='Età %{x}<br>Fondo Pensione: €%{y:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Composizione del Patrimonio nel Tempo (Valori Nominali)',
        xaxis_title="Età",
        yaxis_title="Patrimonio Nominale (€)",
        yaxis_tickformat="€,d",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig

def plot_income_composition(dati_tabella, anni_totali, eta_iniziale):
    """Crea un grafico che mostra la composizione del reddito nel tempo."""
    fig = go.Figure()
    
    anni_asse_x = eta_iniziale + np.arange(1, anni_totali + 1)  # Escludiamo l'anno 0
    
    # Estrai i dati di reddito usando i valori reali calcolati dall'engine
    prelievi_reali = dati_tabella.get('prelievi_effettivi_reali', np.zeros(anni_totali))
    pensioni_reali = dati_tabella.get('pensioni_pubbliche_reali', np.zeros(anni_totali))
    rendite_fp_reali = dati_tabella.get('rendite_fp_reali', np.zeros(anni_totali))
    
    # Crea il grafico a area stack
    fig.add_trace(go.Scatter(
        x=anni_asse_x, y=prelievi_reali, mode='lines',
        stackgroup='one', # Imposta lo stack group per un corretto stacking
        name='Prelievi dal Patrimonio',
        line={'color': '#dc3545'},
        hovertemplate='Età %{x}<br>Prelievi: €%{y:,.0f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=anni_asse_x, y=pensioni_reali, mode='lines',
        stackgroup='one',
        name='Pensione Pubblica',
        line={'color': '#28a745'},
        hovertemplate='Età %{x}<br>Pensione: €%{y:,.0f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=anni_asse_x, y=rendite_fp_reali, mode='lines',
        stackgroup='one',
        name='Rendita Fondo Pensione',
        line={'color': '#ffc107'},
        hovertemplate='Età %{x}<br>Rendita FP: €%{y:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Composizione del Reddito Annuo nel Tempo (Valori Reali)',
        xaxis_title="Età",
        yaxis_title="Reddito Annuo Reale (€)",
        yaxis_tickformat="€,d",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig

if 'simulazione_eseguita' not in st.session_state:
    st.session_state['simulazione_eseguita'] = False
    st.session_state['risultati'] = {}
    st.session_state['parametri'] = {}

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

st.title("✈️ Progetta la Tua Indipendenza Finanziaria")
st.markdown("Benvenuto nel simulatore. Utilizza i controlli nella barra laterale per configurare e lanciare la tua simulazione finanziaria e scoprire come raggiungere i tuoi obiettivi.")

st.sidebar.header("Configurazione Simulazione")

with st.sidebar.expander("📚 Storico Simulazioni", expanded=False):
    saved_simulations = load_simulation_files()
    if not saved_simulations:
        st.caption("Nessuna simulazione salvata.")
    else:
        for sim in saved_simulations:
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.markdown(f"**{sim}**")
                st.caption(f"Salvata il: {datetime.fromtimestamp(os.path.getmtime(os.path.join('simulation_history', sim))).strftime('%d/%m/%Y %H:%M')}")
            with col2:
                if st.button(f"🗑️ Elimina", key=f"del_{sim}"):
                    os.remove(os.path.join('simulation_history', sim))
                    st.rerun()

            with col3:
                if st.button(f"Carica", key=f"load_{sim}"):
                    data = load_simulation_data(sim)
                    st.session_state.parametri = data['parameters']
                    st.session_state.risultati = data['results']
                    st.session_state.simulazione_eseguita = True
                    st.rerun()

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

with st.sidebar.expander("2. Costruttore di Portafoglio ETF", expanded=True):
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
        st.warning(f"L'allocazione totale è {total_allocation:.2f}%. Assicurati che sia 100%.")
    else:
        st.success("Allocazione totale: 100%.")
    
    st.session_state.portfolio = edited_portfolio
    
    weights = edited_portfolio["Allocazione (%)"] / 100
    rendimento_medio_portfolio = np.sum(weights * edited_portfolio["Rendimento Atteso (%)"]) / 100
    volatilita_portfolio = np.sum(weights * edited_portfolio["Volatilità Attesa (%)"]) / 100
    ter_etf_portfolio = np.sum(weights * edited_portfolio["TER (%)"]) / 100

    st.markdown("---")
    st.markdown("##### Parametri Calcolati dal Portafoglio:")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rendimento Medio", f"{rendimento_medio_portfolio:.2%}")
    col2.metric("Volatilità Attesa", f"{volatilita_portfolio:.2%}")
    col3.metric("TER Ponderato", f"{ter_etf_portfolio:.4%}")
    st.caption("La volatilità aggregata è una media ponderata semplificata.")

with st.sidebar.expander("3. Strategie di Prelievo", expanded=True):
    p = st.session_state.get('parametri', {})
    strategia_prelievo = st.selectbox(
        "Strategia di Prelievo",
        options=['FISSO', 'REGOLA_4_PERCENTO', 'GUARDRAIL'],
        index=['FISSO', 'REGOLA_4_PERCENTO', 'GUARDRAIL'].index(p.get('strategia_prelievo', 'REGOLA_4_PERCENTO')),
        help="Scegli come verranno calcolati i prelievi dal tuo patrimonio una volta in pensione. 'FISSO' è un importo costante. 'REGOLA_4_PERCENTO' ricalcola ogni anno il 4% del capitale residuo. 'GUARDRAIL' adatta i prelievi ai trend di mercato per proteggere il capitale."
    )
    prelievo_annuo = st.number_input(
        "Importo Prelievo Fisso Annuo (€)",
        min_value=0, step=500, value=p.get('prelievo_annuo', 12000),
        help="Usato SOLO con la strategia 'FISSO'. Imposta un importo specifico o lascia a 0 per far calcolare al simulatore un prelievo sostenibile. Questo calcolo mira a trovare un importo che il tuo patrimonio possa sostenere per tutta la durata della simulazione con un'alta probabilità di successo, il che potrebbe risultare in un capitale residuo alla fine."
    )
    percentuale_regola_4 = st.slider(
        "Percentuale Regola 4% / Prelievo Iniziale (%)", 0.0, 10.0, p.get('percentuale_regola_4', 0.04) * 100, 0.1,
        help="Il tasso di prelievo iniziale per le strategie 'REGOLA_4_PERCENTO' e 'GUARDRAIL'. Il 4% è una regola standard, ma puoi adattarla alla tua situazione."
    ) / 100
    banda_guardrail = st.slider(
        "Banda Guardrail (%)", 0.0, 50.0, p.get('banda_guardrail', 0.10) * 100, 1.0,
        help="Solo per 'GUARDRAIL'. Se il mercato va molto bene o molto male, questa banda determina se aumentare o diminuire i prelievi per proteggere il capitale o realizzare profitti. Un valore del 10-20% è tipico."
    ) / 100

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

with st.sidebar.expander("5. Tassazione e Costi (Italia)"):
    p = st.session_state.get('parametri', {})
    tassazione_capital_gain = st.slider("Tassazione Capital Gain (%)", 0.0, 50.0, p.get('tassazione_capital_gain', 0.26) * 100, 1.0, help="L'aliquota applicata ai profitti derivanti dalla vendita di ETF. In Italia è tipicamente il 26%.") / 100
    imposta_bollo_titoli = st.slider("Imposta di Bollo Titoli (annua, %)", 0.0, 1.0, p.get('imposta_bollo_titoli', 0.002) * 100, 0.01, help="Tassa patrimoniale annuale sul valore totale del tuo portafoglio titoli. In Italia è lo 0,2%.") / 100
    imposta_bollo_conto = st.number_input("Imposta di Bollo Conto (>5k€)", min_value=0, value=p.get('imposta_bollo_conto', 34), help="Imposta fissa annuale sui conti correnti con giacenza media superiore a 5.000€. In Italia è 34,20€.")
    costo_fisso_etf_mensile = st.number_input("Costo Fisso Deposito Titoli (€/mese)", min_value=0.0, value=p.get('costo_fisso_etf_mensile', 0.0), step=0.5, help="Eventuali costi fissi mensili o annuali addebitati dal tuo broker per il mantenimento del conto titoli. Molti broker online non hanno costi fissi.")

with st.sidebar.expander("6. Fondo Pensione"):
    p = st.session_state.get('parametri', {})
    attiva_fondo_pensione = st.checkbox("Attiva Fondo Pensione", value=p.get('attiva_fondo_pensione', True))
    contributo_annuo_fp = st.number_input("Contributo Annuo FP (€)", min_value=0, step=100, value=p.get('contributo_annuo_fp', 3000), disabled=not attiva_fondo_pensione)
    rendimento_medio_fp = st.slider("Rendimento Medio Annuo FP (%)", 0.0, 15.0, p.get('rendimento_medio_fp', 0.04) * 100, 0.5, disabled=not attiva_fondo_pensione) / 100
    volatilita_fp = st.slider("Volatilità Annuo FP (%)", 0.0, 30.0, p.get('volatilita_fp', 0.08) * 100, 0.5, disabled=not attiva_fondo_pensione) / 100
    ter_fp = st.slider("Costo Annuo (TER) FP (%)", 0.0, 3.0, p.get('ter_fp', 0.01) * 100, 0.1, disabled=not attiva_fondo_pensione) / 100
    tassazione_rendimenti_fp = st.slider("Tassazione Rendimenti FP (%)", 0.0, 30.0, p.get('tassazione_rendimenti_fp', 0.20) * 100, 1.0, disabled=not attiva_fondo_pensione) / 100
    aliquota_finale_fp = st.slider("Aliquota Finale Ritiro FP (%)", 9.0, 23.0, p.get('aliquota_finale_fp', 0.15) * 100, 0.5, disabled=not attiva_fondo_pensione, help="La tassazione agevolata applicata al momento del ritiro del capitale o della rendita dal fondo pensione. Varia dal 15% al 9% in base agli anni di contribuzione.") / 100
    eta_ritiro_fp = st.number_input("Età Ritiro Fondo Pensione", min_value=50, max_value=80, value=p.get('eta_ritiro_fp', 67), disabled=not attiva_fondo_pensione, help="L'età in cui maturi i requisiti per accedere al tuo fondo pensione.")
    percentuale_capitale_fp = st.slider("% Ritiro in Capitale FP", 0.0, 100.0, p.get('percentuale_capitale_fp', 0.50) * 100, 1.0, help="La parte del montante finale che desideri ritirare subito come capitale tassato. Il resto verrà convertito in una rendita mensile.", disabled=not attiva_fondo_pensione) / 100
    durata_rendita_fp_anni = st.number_input("Durata Rendita FP (Anni)", min_value=1, value=p.get('durata_rendita_fp_anni', 25), disabled=not attiva_fondo_pensione, help="Per quanti anni vuoi che venga erogata la rendita calcolata dal tuo fondo pensione.")

with st.sidebar.expander("7. Altre Entrate"):
    p = st.session_state.get('parametri', {})
    pensione_pubblica_annua = st.number_input("Pensione Pubblica Annua (€)", min_value=0, step=500, value=p.get('pensione_pubblica_annua', 8400), help="L'importo annuo lordo della pensione statale (es. INPS) che prevedi di ricevere.")
    inizio_pensione_anni = st.number_input("Inizio Pensione (Anni da oggi)", min_value=0, value=p.get('inizio_pensione_anni', 40), help="Tra quanti anni inizierai a ricevere la pensione pubblica.")

if st.sidebar.button("🚀 Esegui Simulazione", type="primary"):
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

        with st.spinner('Simulazione in corso... Questo potrebbe richiedere qualche istante.'):
            try:
                # Pulisce i risultati precedenti prima di una nuova simulazione
                if 'risultati' in st.session_state:
                    del st.session_state.risultati
                
                # Esegui la simulazione. La nuova logica in `run_full_simulation`
                # gestirà automaticamente il calcolo del prelievo sostenibile se necessario.
                st.session_state.risultati = engine.run_full_simulation(st.session_state.parametri)
                
                st.success('Simulazione completata con successo!')
                # Un piccolo trucco per "pulire" i parametri ?run=... dall'URL dopo la prima esecuzione
                st.query_params.clear()
            except ValueError as e:
                st.error(f"Errore nei parametri: {e}")
            except Exception as e:
                st.error(f"Si è verificato un errore inaspettato durante la simulazione: {e}")

# --- CONTROLLO DI COMPATIBILITÀ DEI RISULTATI ---
# Se i risultati in sessione non sono aggiornati con le nuove chiavi (es. dopo un aggiornamento del codice),
# li cancello per forzare l'utente a rieseguire la simulazione, evitando errori.
if 'risultati' in st.session_state:
    if 'statistiche' not in st.session_state.risultati or \
       'guadagni_accumulo_mediano_nominale' not in st.session_state.risultati['statistiche']:
        del st.session_state.risultati
        st.warning("⚠️ La struttura dei dati è cambiata con l'ultimo aggiornamento. Per favore, clicca di nuovo su 'Avvia Simulazione' per ricalcolare i risultati con la nuova logica.")
        st.stop()

# --- SEZIONE VISUALIZZAZIONE RISULTATI ---
if 'risultati' in st.session_state:
    st.markdown("---")
    st.header("Destino del Tuo Patrimonio: Scenari Possibili")

    stats = st.session_state.risultati['statistiche']
    params = st.session_state.parametri

    with st.expander("💾 Salva Risultati Simulazione"):
        simulation_name = st.text_input("Dai un nome a questa simulazione", f"Simulazione del {datetime.now().strftime('%d/%m/%Y %H:%M')}")
        if st.button("Salva Simulazione"):
            save_simulation(simulation_name, params, st.session_state.risultati)

    st.header("Riepilogo Statistico Chiave")

    patrimonio_iniziale_totale = params['capitale_iniziale'] + params['etf_iniziale']
    contributi_versati = stats['contributi_totali_versati_mediano_nominale']
    guadagni_da_investimento = stats['guadagni_accumulo_mediano_nominale']
    
    reddito_annuo_reale_pensione = st.session_state.risultati['statistiche_prelievi']['totale_reale_medio_annuo']
    patrimonio_finale_reale = stats['patrimonio_finale_mediano_reale']
    anni_di_spesa_coperti = 0.0
    if reddito_annuo_reale_pensione > 0:
        anni_di_spesa_coperti = patrimonio_finale_reale / reddito_annuo_reale_pensione
    else:
        # Se non c'è reddito annuo, il patrimonio non può coprire "anni di spesa"
        anni_di_spesa_coperti = 0.0

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
        help="La ricchezza generata dai soli rendimenti di mercato, calcolata ESCLUSIVAMENTE durante la fase di accumulo (fino all'inizio dei prelievi). Questo valore non tiene conto dei prelievi fatti in pensione."
    )
    col4.metric(
        "Patrimonio Finale in Anni di Spesa", f"{anni_di_spesa_coperti:,.1f} Anni",
        help=f"Il tuo patrimonio finale reale mediano, tradotto in quanti anni del tuo tenore di vita pensionistico (€{reddito_annuo_reale_pensione:,.0f}/anno) può coprire."
    )

    # --- Messaggio speciale per il calcolo del prelievo sostenibile ---
    prelievo_sostenibile_calcolato = stats.get('prelievo_sostenibile_calcolato')
    if prelievo_sostenibile_calcolato is not None:
        st.info(f"""
        **Hai richiesto il calcolo del prelievo massimo sostenibile.**
        
        Abbiamo calcolato che potresti prelevare circa **€ {prelievo_sostenibile_calcolato:,.0f} reali all'anno**.
        
        I risultati della simulazione qui sotto (es. Probabilità di Fallimento) rappresentano uno **stress test** di questo piano. 
        Se la probabilità di fallimento è alta, significa che, a causa della volatilità dei mercati, questo livello di prelievo è considerato rischioso.
        """)

    st.markdown("---")
    
    st.header("Risultati Finali della Simulazione (Patrimonio Nominale)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Patrimonio Finale Mediano (50°)", f"€ {stats['patrimonio_finale_mediano_nominale']:,.0f}")
    col2.metric("Patrimonio Finale (Top 10% - 90°)", f"€ {stats['patrimonio_finale_top_10_nominale']:,.0f}")
    col3.metric("Patrimonio Finale (Peggior 10% - 10°)", f"€ {stats['patrimonio_finale_peggior_10_nominale']:,.0f}")
    col4.metric("Patrimonio Reale Finale Mediano (50°)", f"€ {stats['patrimonio_finale_mediano_reale']:,.0f}")

    st.markdown("##### Indicatori di Rischio del Piano")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Probabilità di Fallimento", f"{stats['probabilita_fallimento']:.2%}", delta=f"{-stats['probabilita_fallimento']:.2%}", delta_color="inverse")
    col2.metric("Drawdown Massimo Peggiore", f"{stats['drawdown_massimo_peggiore']:.2%}", delta=f"{stats['drawdown_massimo_peggiore']:.2%}", delta_color="inverse")
    col3.metric("Sharpe Ratio Medio", f"{stats['sharpe_ratio_medio']:.2f}")

    st.header("Riepilogo Entrate in Pensione (Valori Reali, Scenario Mediano)")
    dati_mediana = st.session_state.risultati['dati_grafici_avanzati']['dati_mediana']
    
    # Calcoliamo le medie direttamente dai dati dello scenario mediano per coerenza
    anni_prelievo_effettivi = np.where(dati_mediana['prelievi_effettivi_reali'] > 0)[0]
    prelievo_medio = np.mean(dati_mediana['prelievi_effettivi_reali'][anni_prelievo_effettivi]) if anni_prelievo_effettivi.size > 0 else 0

    anni_pensione_effettivi = np.where(dati_mediana['pensioni_pubbliche_reali'] > 0)[0]
    pensione_media = np.mean(dati_mediana['pensioni_pubbliche_reali'][anni_pensione_effettivi]) if anni_pensione_effettivi.size > 0 else 0

    anni_rendita_fp_effettivi = np.where(dati_mediana['rendite_fp_reali'] > 0)[0]
    rendita_fp_media = np.mean(dati_mediana['rendite_fp_reali'][anni_rendita_fp_effettivi]) if anni_rendita_fp_effettivi.size > 0 else 0
    
    totale_medio = prelievo_medio + pensione_media + rendita_fp_media

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Prelievo Medio dal Patrimonio", f"€ {prelievo_medio:,.0f}")
    col2.metric("Pensione Pubblica Annua", f"€ {pensione_media:,.0f}")
    col3.metric("Rendita Media da Fondo Pensione", f"€ {rendita_fp_media:,.0f}")
    col4.metric("TOTALE ENTRATE MEDIE ANNUE", f"€ {totale_medio:,.0f}")

    with st.expander("🐞 DEBUG: Dati Grezzi Simulazione"):
        st.write("Array dei patrimoni finali reali (tutte le simulazioni):")
        st.write(st.session_state.risultati['statistiche']['patrimoni_reali_finali'])

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
        - **Le tasse sono semplificate:** Il modello usa un'aliquota fissa sul capital gain, senza considerare scaglioni, minusvalenze pregresse o altre ottimizzazioni fiscali complesse.

        Usa questo strumento come una mappa per definire la direzione, non come un GPS che prevede la destinazione al centimetro.
        """)

    st.header("Analisi Dettagliata per Fasi")
    tab_accumulo, tab_decumulo, tab_dettaglio = st.tabs(["📊 Fase di Accumulo", "🏖️ Fase di Decumulo (Pensione)", "🧾 Dettaglio Flussi di Cassa (Mediano)"])

    with tab_accumulo:
        eta_pensionamento = params['eta_iniziale'] + params['anni_inizio_prelievo']
        st.subheader(f"Dall'età attuale ({params['eta_iniziale']} anni) fino alla pensione (a {eta_pensionamento} anni)")
        st.markdown("In questa fase, i tuoi sforzi si concentrano sulla **costruzione del patrimonio**. I tuoi contributi mensili e annuali, uniti ai rendimenti composti degli investimenti, lavorano insieme per far crescere il capitale che ti sosterrà in futuro.")
        st.markdown("---")
        
        dati_grafici = st.session_state.risultati['dati_grafici_principali']
        
        st.subheader("Evoluzione Patrimonio Reale (Tutti gli Scenari)")
        st.markdown("""
        Questo primo grafico ti dà una visione d'insieme, un "**cono di probabilità**". Non mostra una singola previsione, ma l'intera gamma di risultati possibili, tenendo conto dell'incertezza dei mercati.
        - **La linea rossa (Mediana):** È lo scenario più probabile (50° percentile). Metà delle simulazioni hanno avuto un risultato migliore, metà peggiore.
        - **Le aree colorate:** Rappresentano gli intervalli di confidenza. L'area più scura (25°-75° percentile) è dove il tuo patrimonio ha una buona probabilità di trovarsi. L'area più chiara (10°-90°) mostra gli scenari più estremi, sia positivi che negativi.
        """)
        fig_reale = plot_wealth_summary_chart(
            data=dati_grafici['reale'], 
            title='Evoluzione Patrimonio Reale (Tutti gli Scenari)', 
            y_title='Patrimonio Reale (€)', 
            anni_totali=params['anni_totali'],
            eta_iniziale=params['eta_iniziale'],
            anni_inizio_prelievo=params['anni_inizio_prelievo']
        )
        fig_reale.add_vline(x=params['eta_iniziale'] + params['anni_inizio_prelievo'], line_width=2, line_dash="dash", line_color="grey", annotation_text="Inizio Prelievi")
        st.plotly_chart(fig_reale, use_container_width=True)
        st.markdown("<div style='text-align: center; font-size: 0.9em; font-style: italic;'>Questo è il grafico della verità. Tiene conto dell'inflazione, mostrando il vero potere d'acquisto.</div>", unsafe_allow_html=True)
        st.markdown("---")

        st.subheader("Evoluzione Patrimonio Nominale (Valori Assoluti)")
        st.markdown("""
        Questo grafico mostra l'evoluzione del patrimonio in **valori nominali** (senza considerare l'inflazione). È utile per vedere la crescita assoluta del capitale, ma ricorda che questi valori non riflettono il vero potere d'acquisto futuro.
        
        - **La linea blu (Mediana):** È lo scenario più probabile in valori nominali.
        - **Le aree colorate:** Mostrano gli intervalli di confidenza per i valori nominali.
        - **Confronto con il grafico reale:** La differenza tra i due grafici ti mostra l'impatto dell'inflazione sul tuo patrimonio.
        """)
        fig_nominale = plot_wealth_summary_chart(
            data=dati_grafici['nominale'], 
            title='Evoluzione Patrimonio Nominale (Tutti gli Scenari)', 
            y_title='Patrimonio Nominale (€)', 
            anni_totali=params['anni_totali'],
            eta_iniziale=params['eta_iniziale'],
            anni_inizio_prelievo=params['anni_inizio_prelievo'],
            color_median='#007bff',
            color_fill='#007bff'
        )
        fig_nominale.add_vline(x=params['eta_iniziale'] + params['anni_inizio_prelievo'], line_width=2, line_dash="dash", line_color="grey", annotation_text="Inizio Prelievi")
        st.plotly_chart(fig_nominale, use_container_width=True)
        st.markdown("<div style='text-align: center; font-size: 0.9em; font-style: italic;'>Questo grafico mostra i valori assoluti, senza considerare l'inflazione.</div>", unsafe_allow_html=True)
        st.markdown("---")

        st.subheader("Quali sono i percorsi possibili? (Visione di dettaglio)")
        st.markdown("""
        Se il grafico precedente era una "mappa meteorologica", questo è come guardare le traiettorie di 50 aerei diversi che volano nella stessa tempesta. Ogni linea colorata è una delle possibili vite del tuo portafoglio tra le migliaia simulate. 
        
        Questo grafico ti aiuta a capire la natura caotica dei mercati: alcuni percorsi sono fortunati (linee che finiscono in alto), altri meno. La **linea rossa più spessa** è sempre lo scenario mediano, il tuo punto di riferimento. Osserva come, nonostante le partenze simili, le traiettorie divergono enormemente nel tempo. Questo è il motivo per cui diversificare e avere un piano a lungo termine è fondamentale.
        """)
        
        fig_spaghetti = plot_spaghetti_chart(
            data=dati_grafici['reale'],
            title="Percorsi Individuali delle Simulazioni (Patrimonio Reale)",
            y_title="Patrimonio Reale (€)",
            anni_totali=params['anni_totali'],
            eta_iniziale=params['eta_iniziale'],
            anni_inizio_prelievo=params['anni_inizio_prelievo']
        )
        st.plotly_chart(fig_spaghetti, use_container_width=True)

        st.markdown("---")
        # SEZIONE TABELLA DETTAGLIATA

    with tab_decumulo:
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
            params['anni_inizio_prelievo'],
            eta_iniziale=params['eta_iniziale']
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
            st.session_state.risultati['statistiche']['patrimoni_reali_finali'],
            dati_principali['reale'],
            params['anni_totali'],
            eta_iniziale=params['eta_iniziale']
        )
        fig_worst.add_vline(x=params['eta_iniziale'] + params['anni_inizio_prelievo'], line_width=2, line_dash="dash", line_color="grey", annotation_text="Inizio Prelievi")
        st.plotly_chart(fig_worst, use_container_width=True)

    with tab_dettaglio:
        st.subheader("Analisi Finanziaria Annuale Dettagliata (Simulazione Mediana)")
        st.markdown("Questa sezione è la 'radiografia' dello scenario mediano (il più probabile). I grafici e la tabella mostrano, anno per anno, tutti i flussi finanziari e l'evoluzione del patrimonio.")
        
        dati_tabella = st.session_state.risultati['dati_grafici_avanzati']['dati_mediana']
        
        st.markdown("---")
        st.subheader("Grafici di Dettaglio (Scenario Mediano)")

        # Grafico 1: Composizione del Patrimonio Nominale
        st.markdown("##### Da cosa è composto il mio patrimonio?")
        st.markdown("Questo grafico mostra come evolvono nel tempo le tre componenti principali del tuo patrimonio: Liquidità, ETF e Fondo Pensione. I valori sono **nominali**, cioè non tengono conto dell'inflazione.")
        fig_composizione_patrimonio = plot_wealth_composition_over_time_nominal(dati_tabella, params['anni_totali'], eta_iniziale=params['eta_iniziale'])
        st.plotly_chart(fig_composizione_patrimonio, use_container_width=True)

        # Grafico 2: Composizione del Reddito Annuo Reale
        st.markdown("##### Da dove arriveranno i miei soldi ogni anno in pensione?")
        st.markdown("Questo grafico è fondamentale: mostra, anno per anno, da quali fonti proverrà il tuo reddito per vivere. I valori sono **reali** (potere d'acquisto di oggi) per darti un'idea concreta del tuo tenore di vita. Puoi vedere come i prelievi dal patrimonio vengono progressivamente sostituiti o integrati da pensione e rendite.")
        fig_composizione_reddito = plot_income_composition(dati_tabella, params['anni_totali'], eta_iniziale=params['eta_iniziale'])
        st.plotly_chart(fig_composizione_reddito, use_container_width=True)

        st.markdown("---")
        st.subheader("Tabella Dettagliata Anno per Anno")
        
        # Costruzione del DataFrame
        num_anni = params['anni_totali']
        df_index = np.arange(1, num_anni + 1)
        
        # Assicuriamoci che tutti gli array siano della lunghezza corretta (num_anni)
        # I dati dall'engine sono indicizzati 0..N, dove l'indice 0 è l'anno 1.
        df_data = {
            'Anno': df_index,
            'Età': params['eta_iniziale'] + df_index
        }
        
        col_keys = [
            ('Obiettivo Prelievo (Nom.)', 'prelievi_target_nominali'),
            ('Prelievo Effettivo (Nom.)', 'prelievi_effettivi_nominali'),
            ('Prelievo Effettivo (Reale)', 'prelievi_effettivi_reali'),
            ('Fonte: Conto Corrente', 'prelievi_da_banca_nominali'),
            ('Fonte: Vendita ETF', 'prelievi_da_etf_nominali'),
            ('Vendita ETF (Rebalance)', 'vendite_rebalance_nominali'),
            ('Pensione Pubblica (Nom.)', 'pensioni_pubbliche_nominali'),
            ('Rendita FP (Nom.)', 'rendite_fp_nominali'),
            ('Liquidazione FP (Nom.)', 'fp_liquidato_nominale'),
            # Per i saldi, partiamo dall'anno 1 per allinearli con gli anni del dataframe
            ('Patrimonio Banca (Nom.)', 'saldo_banca_nominale'),
            ('Patrimonio ETF (Nom.)', 'saldo_etf_nominale'),
            ('Patrimonio FP (Nom.)', 'saldo_fp_nominale'),
            ('Patrimonio Banca (Reale)', 'saldo_banca_reale')
        ]

        for col, key in col_keys:
            full_array = dati_tabella.get(key, np.zeros(num_anni + 1))
            if 'saldo' in key:
                 # I saldi includono l'anno 0, quindi prendiamo gli elementi da 1 a num_anni
                df_data[col] = full_array[1:num_anni+1]
            else:
                # I flussi sono calcolati per gli anni 1..num_anni, quindi prendiamo i primi num_anni elementi
                df_data[col] = full_array[:num_anni]
        
        df = pd.DataFrame(df_data)
        
        st.dataframe(df.style.format({
            'Obiettivo Prelievo (Nom.)': "€ {:,.0f}",
            'Prelievo Effettivo (Nom.)': "€ {:,.0f}",
            'Fonte: Conto Corrente': "€ {:,.0f}",
            'Fonte: Vendita ETF': "€ {:,.0f}",
            'Vendita ETF (Rebalance)': "€ {:,.0f}",
            'Prelievo Effettivo (Reale)': "€ {:,.0f}",
            'Pensione Pubblica (Nom.)': "€ {:,.0f}",
            'Rendita FP (Nom.)': "€ {:,.0f}",
            'Liquidazione FP (Nom.)': "€ {:,.0f}",
            'Patrimonio Banca (Nom.)': "€ {:,.0f}",
            'Patrimonio ETF (Nom.)': "€ {:,.0f}",
            'Patrimonio FP (Nom.)': "€ {:,.0f}",
            'Patrimonio Banca (Reale)': "€ {:,.0f}",
        }))

        with st.expander("Guida alla Lettura della Tabella"):
            st.markdown("""
            - **Obiettivo Prelievo vs Prelievo Effettivo**: L''Obiettivo' è quanto vorresti prelevare. L''Effettivo' è quanto prelevi realmente. Se hai pochi soldi, l''Effettivo' sarà più basso.
            - **Fonte Conto vs Fonte ETF**: Mostrano da dove provengono i soldi per il prelievo. Prima si usa la liquidità sul conto, poi si vendono gli ETF.
            - **Vendita ETF (Rebalance)**: NON sono soldi spesi. Sono vendite fatte per ridurre il rischio (seguendo il Glidepath). I soldi vengono spostati da ETF a liquidità.
            - **Liquidazione Capitale FP**: Somma che ricevi tutta in una volta dal fondo pensione all'età scelta. Aumenta di molto la tua liquidità in quell'anno.
            - **Entrate Anno (Reali)**: La somma di tutte le tue entrate (prelievi, pensioni) in potere d'acquisto di oggi. Questa cifra misura il tuo vero tenore di vita annuale.
            """) 