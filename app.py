import streamlit as st
import numpy as np
import simulation_engine as engine
import plotly.graph_objects as go
import pandas as pd
import json
import os
from datetime import datetime
import plotly.express as px

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

def plot_percentile_chart(data, title, y_title, color_median, color_fill, anni_totali, eta_iniziale):
    """Crea un grafico a 'cono' con i percentili."""
    fig = go.Figure()
    anni_asse_x = eta_iniziale + np.linspace(0, anni_totali, data.shape[1])

    p10 = np.percentile(data, 10, axis=0)
    p25 = np.percentile(data, 25, axis=0)
    median_data = np.median(data, axis=0)
    p75 = np.percentile(data, 75, axis=0)
    p90 = np.percentile(data, 90, axis=0)

    rgb_fill = hex_to_rgb(color_fill)

    fig.add_trace(go.Scatter(
        x=np.concatenate([anni_asse_x, anni_asse_x[::-1]]),
        y=np.concatenate([p90, p10[::-1]]),
        fill='toself',
        fillcolor=f'rgba({rgb_fill[0]}, {rgb_fill[1]}, {rgb_fill[2]}, 0.2)',
        line={'color': 'rgba(255,255,255,0)'},
        name='10-90 Percentile',
        hoverinfo='none'
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([anni_asse_x, anni_asse_x[::-1]]),
        y=np.concatenate([p75, p25[::-1]]),
        fill='toself',
        fillcolor=f'rgba({rgb_fill[0]}, {rgb_fill[1]}, {rgb_fill[2]}, 0.4)',
        line={'color': 'rgba(255,255,255,0)'},
        name='25-75 Percentile',
        hoverinfo='none'
    ))
    fig.add_trace(go.Scatter(
        x=anni_asse_x, y=median_data, mode='lines',
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

def plot_spaghetti_chart(data, title, y_title, color_median, anni_totali, anni_inizio_prelievo, eta_iniziale):
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
            name=f'Simulazione {i}' # Nome univoco per ogni traccia
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

def plot_income_composition(data, anni_totali, eta_iniziale):
    """Crea un grafico ad area della composizione del reddito annuo reale."""
    # L'asse X va da 1 ad anni_totali, perchè il reddito è un flusso annuale
    anni_asse_x_annuale = eta_iniziale + np.arange(1, anni_totali + 1)
    fig = go.Figure()
    
    # Usiamo :anni_totali per prendere i primi N anni di flussi, escludendo l'ultimo punto non calcolato
    fig.add_trace(go.Scatter(
        x=anni_asse_x_annuale, y=data['prelievi_effettivi_reali'][:anni_totali],
        name='Prelievi dal Patrimonio', stackgroup='one',
        line={'color': '#4472C4'}
    ))
    fig.add_trace(go.Scatter(
        x=anni_asse_x_annuale, y=data['pensioni_pubbliche_reali'][:anni_totali],
        name='Pensione Pubblica', stackgroup='one',
        line={'color': '#ED7D31'}
    ))
    fig.add_trace(go.Scatter(
        x=anni_asse_x_annuale, y=data['rendite_fp_reali'][:anni_totali],
        name='Rendita Fondo Pensione', stackgroup='one',
        line={'color': '#A5A5A5'}
    ))

    fig.update_layout(
        title='Composizione del Reddito Annuo Reale (Scenario Mediano)',
        xaxis_title="Età",
        yaxis_title='Reddito Reale Annuo (€ Odierni)',
        yaxis_tickformat="€,d",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig

def plot_wealth_composition_over_time_nominal(data, anni_totali, eta_iniziale):
    """Crea un grafico stacked area per la composizione del patrimonio nominale nel tempo."""
    anni_asse_x_annuale = eta_iniziale + np.arange(anni_totali + 1)
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=anni_asse_x_annuale, y=data['saldo_banca_nominale'],
        name='Liquidità (Conto Corrente)', stackgroup='one',
        line={'color': '#5B9BD5'}
    ))
    fig.add_trace(go.Scatter(
        x=anni_asse_x_annuale, y=data['saldo_etf_nominale'],
        name='Portafoglio ETF', stackgroup='one',
        line={'color': '#ED7D31'}
    ))
    fig.add_trace(go.Scatter(
        x=anni_asse_x_annuale, y=data['saldo_fp_nominale'],
        name='Fondo Pensione', stackgroup='one',
        line={'color': '#70AD47'}
    ))

    fig.update_layout(
        title='Composizione del Patrimonio Nominale nel Tempo (Scenario Mediano)',
        xaxis_title="Età",
        yaxis_title='Patrimonio Nominale (€)',
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

def plot_income_cone_chart(data, anni_totali, anni_inizio_prelievo, eta_iniziale):
    """Crea un grafico a 'cono' per il reddito reale annuo."""
    fig = go.Figure()
    start_index = int(anni_inizio_prelievo)
    
    # La simulazione dura N anni, quindi i dati sono calcolati per gli anni da 0 a N-1.
    # Il grafico deve quindi mostrare i dati fino all'anno N-1.
    end_index = int(anni_totali)

    if start_index >= end_index:
        # Non c'è un periodo di decumulo da mostrare
        return fig 

    anni_asse_x = eta_iniziale + np.arange(start_index, end_index) # Es: da 35 a 79 se anni_totali=80
    data_decumulo = data[:, start_index:end_index] # Seleziona le colonne corrispondenti

    p10 = np.percentile(data_decumulo, 10, axis=0)
    p25 = np.percentile(data_decumulo, 25, axis=0)
    median_data = np.median(data_decumulo, axis=0)
    p75 = np.percentile(data_decumulo, 75, axis=0)
    p90 = np.percentile(data_decumulo, 90, axis=0)

    fig.add_trace(go.Scatter(
        x=np.concatenate([anni_asse_x, anni_asse_x[::-1]]),
        y=np.concatenate([p90, p10[::-1]]),
        fill='toself',
        fillcolor='rgba(0, 176, 246, 0.2)',
        line={'color': 'rgba(255,255,255,0)'},
        name='10-90 Percentile',
        hoverinfo='none'
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([anni_asse_x, anni_asse_x[::-1]]),
        y=np.concatenate([p75, p25[::-1]]),
        fill='toself',
        fillcolor='rgba(0, 176, 246, 0.4)',
        line={'color': 'rgba(255,255,255,0)'},
        name='25-75 Percentile',
        hoverinfo='none'
    ))
    fig.add_trace(go.Scatter(
        x=anni_asse_x, y=median_data, mode='lines',
        name='Reddito Mediano (50°)',
        line={'width': 3, 'color': '#00B0F0'},
        hovertemplate='Età %{x}<br>Reddito Mediano: €%{y:,.0f}<extra></extra>'
    ))

    fig.update_layout(
        title="Quale Sarà il Mio Tenore di Vita? (Reddito Annuo Reale)",
        xaxis_title="Età",
        yaxis_title="Reddito Annuo Reale (€ di oggi)",
        yaxis_tickformat="€,d",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig

def plot_wealth_composition_chart(initial, contributions, gains):
    """Crea un grafico a barre per la composizione della ricchezza finale."""
    labels = ['Patrimonio Iniziale', 'Contributi Versati', 'Guadagni da Investimento']
    values = [initial, contributions, gains]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

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

def plot_worst_scenarios_chart(patrimoni_finali, data, anni_totali, eta_iniziale):
    """Mostra un'analisi degli scenari peggiori (es. 10% dei casi)."""
    
    # FIX: Se non ci sono scenari di successo (es. fallimento 100%), non possiamo
    # analizzare i "peggiori tra i successi". Mostriamo un messaggio all'utente.
    if patrimoni_finali.size == 0:
        fig = go.Figure()
        fig.add_annotation(text="Probabilità di fallimento del 100%: non ci sono scenari di successo da analizzare.",
                           xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=16))
        fig.update_layout(
            xaxis_visible=False,
            yaxis_visible=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        return fig

    fig = go.Figure()

    soglia_peggiore = np.percentile(patrimoni_finali, 10)
    indici_peggiori = np.where(patrimoni_finali <= soglia_peggiore)[0]
    
    anni_asse_x = eta_iniziale + np.linspace(0, anni_totali, data.shape[1])
    
    if len(indici_peggiori) > 0:
        indici_da_mostrare = np.random.choice(indici_peggiori, size=min(50, len(indici_peggiori)), replace=False)
        
        for i in indici_da_mostrare:
            fig.add_trace(go.Scatter(
                x=anni_asse_x, y=data[i, :], mode='lines',
                line={'width': 1, 'color': 'rgba(255, 82, 82, 0.5)'},
                hoverinfo='none', showlegend=False
            ))

    mediana_scenari_peggiori = np.median(data[indici_peggiori, :], axis=0) if len(indici_peggiori) > 0 else np.zeros(data.shape[1])
    fig.add_trace(go.Scatter(
        x=anni_asse_x, y=mediana_scenari_peggiori, mode='lines',
        name='Mediana Scenari Peggiori',
        line={'width': 2.5, 'color': '#FF5252'},
        hovertemplate='Età %{x:.1f}<br>Patrimonio: €%{y:,.0f}<extra></extra>'
    ))
            
    fig.update_layout(
        title="Come si Comporta il Piano negli Scenari Peggiori? (Analisi del Rischio)",
        xaxis_title="Età",
        yaxis_title="Patrimonio Reale (€ di oggi)",
        yaxis_tickformat="€,d",
        hovermode="x unified",
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )
    return fig

# --- INIZIALIZZAZIONE ROBUSTA DELLO STATO ---
def initialize_session_state():
    """
    Inizializza lo stato della sessione con tutti i parametri di default
    per evitare KeyError al primo avvio o dopo un refresh.
    """
    default_params = {
        # Impostazioni Generali
        'eta_iniziale': 30,
        'capitale_iniziale': 10000,
        'etf_iniziale': 5000,
        'contributo_mensile_banca': 200,
        'contributo_mensile_etf': 800,
        'anni_totali': 70,
        'inflazione': 0.02,
        'n_simulazioni': 1000,
        
        # Strategia di Prelievo
        'strategia_prelievo': 'FISSO',
        'prelievo_annuo': 0,
        'percentuale_regola_4': 0.04,
        'banda_guardrail': 0.10,
        'anni_inizio_prelievo': 35,

        # Glidepath
        'attiva_glidepath': True,
        'inizio_glidepath_anni': 25,
        'fine_glidepath_anni': 35,
        'allocazione_etf_finale': 0.20,

        # Tassazione e Costi
        'tassazione_capital_gain': 0.26,
        'imposta_bollo_titoli': 0.002,
        'imposta_bollo_conto': 34.20,
        'costo_fisso_etf_mensile': 0,

        # Fondo Pensione
        'attiva_fondo_pensione': False,
        'contributo_annuo_fp': 0,
        'rendimento_medio_fp': 0.05,
        'volatilita_fp': 0.08,
        'ter_fp': 0.01,
        'tassazione_rendimenti_fp': 0.20,
        'aliquota_finale_fp': 0.15,
        'eta_ritiro_fp': 67,
        'percentuale_capitale_fp': 0.50,
        'durata_rendita_fp_anni': 25,

        # Pensione Statale
        'pensione_pubblica_annua': 20000,
        'inizio_pensione_anni': 30,
    }

    if 'parametri' not in st.session_state:
        st.session_state.parametri = default_params

    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = pd.DataFrame({
            "Asset Class": ["Azionario Globale", "Obbligazionario Globale"],
            "Allocazione (%)": [80.0, 20.0],
            "Rendimento Atteso (%)": [8.5, 3.0],
            "Volatilità Attesa (%)": [16.0, 6.0],
            "TER (%)": [0.20, 0.15]
        })

# Chiamiamo la funzione di inizializzazione all'inizio
initialize_session_state()

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(
    page_title="Progetta il Tuo Futuro Finanziario",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Progetta la Tua Indipendenza Finanziaria")
st.markdown("Benvenuto nel simulatore. Utilizza i controlli nella barra laterale per configurare e lanciare la tua simulazione finanziaria e scoprire come raggiungere i tuoi obiettivi.")

st.sidebar.header("Configurazione Simulazione")

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
                if st.button(f"🗑️ Elimina", key=f"del_{sim['filename']}"):
                    delete_simulation(sim['filename'])
                    st.rerun()

            with col3:
                if st.button(f"Carica", key=f"load_{sim['filename']}"):
                    data = load_simulation_data(sim['filename'])
                    st.session_state.parametri = data['parameters']
                    st.session_state.risultati = data['results']
                    st.session_state.simulazione_eseguita = True
                    st.rerun()

# --- Sezione 1: Patrimonio e Contributi ---
with st.sidebar.expander("1. Patrimonio e Contributi", expanded=True):
    st.session_state.parametri['capitale_iniziale'] = st.number_input(
        "Capitale Conto Corrente (€)", min_value=0, step=1000,
        value=st.session_state.parametri.get('capitale_iniziale', 10000),
        help="La liquidità che hai oggi sul conto corrente o in asset a bassissimo rischio/rendimento."
    )
    st.session_state.parametri['etf_iniziale'] = st.number_input(
        "Valore Portafoglio ETF (€)", min_value=0, step=1000,
        value=st.session_state.parametri.get('etf_iniziale', 5000),
        help="Il valore di mercato attuale di tutti i tuoi investimenti in ETF/azioni."
    )
    st.session_state.parametri['contributo_mensile_banca'] = st.number_input(
        "Contributo Mensile Conto (€)", min_value=0, step=50,
        value=st.session_state.parametri.get('contributo_mensile_banca', 200),
        help="La cifra che riesci a risparmiare e accantonare sul conto corrente ogni mese."
    )
    st.session_state.parametri['contributo_mensile_etf'] = st.number_input(
        "Contributo Mensile ETF (€)", min_value=0, step=50,
        value=st.session_state.parametri.get('contributo_mensile_etf', 800),
        help="La cifra che investi attivamente ogni mese nel tuo portafoglio ETF (PAC)."
    )

# --- Sezione 2: Costruttore di Portafoglio ---
def get_portfolio_summary():
    """Calcola rendimento, volatilità e TER ponderati del portafoglio."""
    portfolio_df = st.session_state.portfolio
    weights = portfolio_df["Allocazione (%)"] / 100
    rendimento_medio = np.sum(weights * portfolio_df["Rendimento Atteso (%)"]) / 100
    volatilita = np.sum(weights * portfolio_df["Volatilità Attesa (%)"]) / 100
    ter_ponderato = np.sum(weights * portfolio_df["TER (%)"]) / 100
    return rendimento_medio, volatilita, ter_ponderato

with st.sidebar.expander("2. Costruttore di Portafoglio ETF", expanded=True):
    st.markdown("Modifica l'allocazione, il TER e le stime di rendimento/volatilità per ogni asset.")
    
    edited_portfolio = st.data_editor(
        st.session_state.portfolio,
        column_config={
            "Allocazione (%)": st.column_config.NumberColumn(format="%.1f%%", min_value=0, max_value=100),
            "TER (%)": st.column_config.NumberColumn(format="%.2f%%", min_value=0),
            "Rendimento Atteso (%)": st.column_config.NumberColumn(format="%.1f%%"),
            "Volatilità Attesa (%)": st.column_config.NumberColumn(format="%.1f%%"),
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
    
    rendimento_medio_p, volatilita_p, ter_p = get_portfolio_summary()

    st.markdown("---")
    st.markdown("##### Parametri Calcolati dal Portafoglio:")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rendimento Medio", f"{rendimento_medio_p:.2%}")
    col2.metric("Volatilità Attesa", f"{volatilita_p:.2%}")
    col3.metric("TER Ponderato", f"{ter_p:.4f}%")
    st.caption("La volatilità aggregata è una media ponderata semplificata.")

with st.sidebar.expander("3. Impostazioni Generali", expanded=False):
    # Usiamo .get() per sicurezza, anche se l'inizializzazione dovrebbe aver già creato la chiave
    st.session_state.parametri['eta_iniziale'] = st.number_input(
        "Età Iniziale", min_value=18, max_value=100, 
        value=st.session_state.parametri.get('eta_iniziale', 30), step=1
    )
    st.session_state.parametri['anni_totali'] = st.number_input(
        "Orizzonte Temporale (Anni)", min_value=1, max_value=100, 
        value=st.session_state.parametri.get('anni_totali', 70), step=1,
        help="L'orizzonte temporale totale della simulazione, in anni. Es. fino a 100 anni di età."
    )
    st.session_state.parametri['inflazione'] = st.slider(
        "Tasso di Inflazione Medio Annuo (%)", 0.0, 10.0, st.session_state.parametri.get('inflazione', 0.02) * 100, 0.5,
        format="%.1f%%",
        help="L'inflazione media attesa per anno. Riduce il potere d'acquisto del tuo patrimonio nel tempo."
    )
    st.session_state.parametri['n_simulazioni'] = st.select_slider(
        "Numero di Simulazioni (Precisione)",
        options=[100, 500, 1000, 2000, 5000],
        value=st.session_state.parametri.get('n_simulazioni', 1000),
        help="Il numero di scenari futuri da simulare. Più alto è il numero, più accurati i risultati ma più lenta la simulazione."
    )

# --- Sezione 4: Strategia di Prelievo ---
with st.sidebar.expander("4. Strategia di Prelievo", expanded=False):
    st.session_state.parametri['strategia_prelievo'] = st.selectbox(
        "Strategia di Prelievo", 
        options=['FISSO', 'REGOLA_4_PERCENTO', 'GUARDRAIL'], 
        index=['FISSO', 'REGOLA_4_PERCENTO', 'GUARDRAIL'].index(st.session_state.parametri.get('strategia_prelievo', 'FISSO')),
        help="Scegli come verranno calcolati i prelievi una volta in pensione. 'FISSO' è un importo costante. 'REGOLA_4_PERCENTO' ricalcola ogni anno il 4% del capitale residuo. 'GUARDRAIL' adatta i prelievi ai trend di mercato per proteggere il capitale."
    )
    
    is_fisso = st.session_state.parametri['strategia_prelievo'] == 'FISSO'
    is_regola4 = st.session_state.parametri['strategia_prelievo'] in ['REGOLA_4_PERCENTO', 'GUARDRAIL']

    st.session_state.parametri['prelievo_annuo'] = st.number_input(
        "Importo Prelievo Fisso Annuo (€)", min_value=0, step=1000, 
        value=st.session_state.parametri.get('prelievo_annuo', 0),
        disabled=not is_fisso,
        help="Usato SOLO con la strategia 'FISSO'. Imposta l'esatto importo lordo che vuoi prelevare ogni anno. Lascia a 0 per far calcolare al simulatore il prelievo massimo sostenibile."
    )
    st.session_state.parametri['percentuale_regola_4'] = st.slider(
        "Percentuale Regola 4% / Prelievo Iniziale (%)", 0.0, 10.0, 
        st.session_state.parametri.get('percentuale_regola_4', 0.04) * 100, 0.5,
        format="%.2f%%",
        disabled=not is_regola4,
        help="La percentuale del patrimonio da prelevare il primo anno (per GUARDRAIL) o ogni anno (per REGOLA 4%)."
    )
    st.session_state.parametri['banda_guardrail'] = st.slider(
        "Banda Guardrail (%)", 0.0, 50.0, 
        st.session_state.parametri.get('banda_guardrail', 0.10) * 100, 1.0,
        format="%.1f%%",
        disabled=st.session_state.parametri['strategia_prelievo'] != 'GUARDRAIL',
        help="I limiti superiore e inferiore (es. +/- 10%) che attivano aggiustamenti al prelievo annuale per proteggere il capitale."
    )
    st.session_state.parametri['anni_inizio_prelievo'] = st.number_input(
        "Anni all'inizio dei prelievi", 0, 
        st.session_state.parametri['anni_totali'], 
        st.session_state.parametri.get('anni_inizio_prelievo', 35), 1,
        help="Tra quanti anni inizierai a prelevare dal tuo patrimonio per le spese di vita."
    )

# --- Sezione 5: Glidepath ---
with st.sidebar.expander("5. Asset Allocation Dinamica (Glidepath)", expanded=False):
    st.session_state.parametri['attiva_glidepath'] = st.checkbox(
        "Attiva Glidepath", 
        value=st.session_state.parametri.get('attiva_glidepath', True),
        help="Attiva un ribilanciamento automatico che riduce gradualmente l'esposizione azionaria con l'avvicinarsi della pensione."
    )
    st.session_state.parametri['inizio_glidepath_anni'] = st.number_input(
        "Inizio Glidepath (Anni da oggi)", 0, 
        st.session_state.parametri['anni_totali'], 
        st.session_state.parametri.get('inizio_glidepath_anni', 25), 1,
        disabled=not st.session_state.parametri['attiva_glidepath'],
        help="L'anno in cui inizia la riduzione progressiva del rischio."
    )
    st.session_state.parametri['fine_glidepath_anni'] = st.number_input(
        "Fine Glidepath (Anni da oggi)", 
        st.session_state.parametri.get('inizio_glidepath_anni', 25), 
        st.session_state.parametri['anni_totali'], 
        st.session_state.parametri.get('fine_glidepath_anni', 35), 1,
        disabled=not st.session_state.parametri['attiva_glidepath'],
        help="L'anno in cui si raggiunge l'allocazione finale, più conservativa."
    )
    st.session_state.parametri['allocazione_etf_finale'] = st.slider(
        "Allocazione ETF Finale (%)", 0.0, 100.0, 
        st.session_state.parametri.get('allocazione_etf_finale', 0.20) * 100, 1.0,
        format="%.0f%%",
        disabled=not st.session_state.parametri['attiva_glidepath'],
        help="La percentuale di ETF (rischiosa) che vuoi avere alla fine del Glidepath. Il resto sarà liquidità."
    )

# --- Sezione 6: Tassazione e Costi ---
with st.sidebar.expander("6. Tassazione e Costi (Italia)", expanded=False):
    st.session_state.parametri['tassazione_capital_gain'] = st.slider(
        "Tassazione Capital Gain (%)", 0.0, 50.0, 
        st.session_state.parametri.get('tassazione_capital_gain', 0.26) * 100, 0.5,
        format="%.1f%%",
        help="L'aliquota applicata ai guadagni derivanti dalla vendita di ETF."
    )
    st.session_state.parametri['imposta_bollo_titoli'] = st.slider(
        "Imposta di Bollo su Titoli (% Annua)", 0.0, 1.0, 
        st.session_state.parametri.get('imposta_bollo_titoli', 0.002) * 100, 0.01,
        format="%.2f%%",
        help="L'imposta di bollo annuale sul valore del tuo portafoglio ETF."
    )
    st.session_state.parametri['imposta_bollo_conto'] = st.number_input(
        "Imposta di Bollo su Conto Corrente (€ Annua)", min_value=0.0, 
        value=st.session_state.parametri.get('imposta_bollo_conto', 34.20), step=1.0,
        help="L'imposta di bollo fissa annuale per giacenze medie superiori a 5.000€."
    )
    st.session_state.parametri['costo_fisso_etf_mensile'] = st.number_input(
        "Costi Fissi Mensili Conto Titoli (€)", min_value=0.0, 
        value=st.session_state.parametri.get('costo_fisso_etf_mensile', 0.0), step=0.5,
        help="Eventuali costi fissi mensili del tuo broker/conto titoli."
    )

# --- Sezione 7: Fondo Pensione ---
with st.sidebar.expander("7. Fondo Pensione", expanded=False):
    st.session_state.parametri['attiva_fondo_pensione'] = st.checkbox(
        "Attiva Fondo Pensione", 
        value=st.session_state.parametri.get('attiva_fondo_pensione', False),
        help="Includi un fondo pensione complementare nella simulazione."
    )
    st.session_state.parametri['contributo_annuo_fp'] = st.number_input(
        "Contributo Annuo al Fondo Pensione (€)", min_value=0, 
        value=st.session_state.parametri.get('contributo_annuo_fp', 0), step=100,
        disabled=not st.session_state.parametri['attiva_fondo_pensione']
    )
    st.session_state.parametri['rendimento_medio_fp'] = st.slider(
        "Rendimento Medio Annuo Lordo FP (%)", 0.0, 15.0, 
        st.session_state.parametri.get('rendimento_medio_fp', 0.05) * 100, 0.5,
        format="%.1f%%",
        disabled=not st.session_state.parametri['attiva_fondo_pensione']
    )
    st.session_state.parametri['volatilita_fp'] = st.slider(
        "Volatilità Attesa FP (%)", 0.0, 30.0, 
        st.session_state.parametri.get('volatilita_fp', 0.08) * 100, 0.5,
        format="%.1f%%",
        disabled=not st.session_state.parametri['attiva_fondo_pensione']
    )
    st.session_state.parametri['ter_fp'] = st.slider(
        "Costo Annuo (TER) FP (%)", 0.0, 5.0, 
        st.session_state.parametri.get('ter_fp', 0.01) * 100, 0.1,
        format="%.2f%%",
        disabled=not st.session_state.parametri['attiva_fondo_pensione']
    )
    st.session_state.parametri['tassazione_rendimenti_fp'] = st.slider(
        "Tassazione Rendimenti FP (%)", 0.0, 30.0, 
        st.session_state.parametri.get('tassazione_rendimenti_fp', 0.20) * 100, 0.5,
        format="%.1f%%",
        disabled=not st.session_state.parametri['attiva_fondo_pensione'],
        help="La tassazione applicata annualmente sui rendimenti maturati nel fondo pensione."
    )
    st.session_state.parametri['aliquota_finale_fp'] = st.slider(
        "Aliquota Fiscale Finale FP (%)", 0.0, 30.0, 
        st.session_state.parametri.get('aliquota_finale_fp', 0.15) * 100, 0.5,
        format="%.1f%%",
        disabled=not st.session_state.parametri['attiva_fondo_pensione'],
        help="L'aliquota fiscale applicata al momento della liquidazione del capitale o della rendita (tassazione agevolata dal 15% al 9%)."
    )
    st.session_state.parametri['eta_ritiro_fp'] = st.number_input(
        "Età Ritiro Capitale/Rendita FP", 
        min_value=st.session_state.parametri['eta_iniziale'], 
        max_value=st.session_state.parametri['eta_iniziale'] + st.session_state.parametri['anni_totali'],
        value=st.session_state.parametri.get('eta_ritiro_fp', 67),
        disabled=not st.session_state.parametri['attiva_fondo_pensione'],
        help="L'età in cui smetti di contribuire e inizi a ricevere le prestazioni dal fondo pensione."
    )
    st.session_state.parametri['percentuale_capitale_fp'] = st.slider(
        "Percentuale Liquidazione Capitale FP (%)", 0.0, 100.0, 
        st.session_state.parametri.get('percentuale_capitale_fp', 0.50) * 100, 5.0,
        format="%.0f%%",
        disabled=not st.session_state.parametri['attiva_fondo_pensione'],
        help="La percentuale del montante che vuoi ritirare subito come capitale. Il resto verrà convertito in rendita."
    )
    st.session_state.parametri['durata_rendita_fp_anni'] = st.number_input(
        "Durata Rendita FP (Anni)", min_value=1, 
        value=st.session_state.parametri.get('durata_rendita_fp_anni', 25),
        disabled=not st.session_state.parametri['attiva_fondo_pensione'],
        help="Per quanti anni vuoi che venga erogata la rendita calcolata dal tuo fondo pensione."
    )

# --- Sezione 8: Pensione Statale ---
with st.sidebar.expander("8. Pensione Statale", expanded=False):
    st.session_state.parametri['pensione_pubblica_annua'] = st.number_input(
        "Pensione Pubblica Annua Lorda Attesa (€)",
        min_value=0,
        value=st.session_state.parametri.get('pensione_pubblica_annua', 20000),
        step=1000,
        help="Inserisci l'importo annuo lordo della tua pensione pubblica (es. INPS) che ti aspetti di ricevere. Il valore verrà adeguato all'inflazione."
    )
    st.session_state.parametri['inizio_pensione_anni'] = st.number_input(
        "Inizio Erogazione Pensione (Anni da oggi)",
        min_value=0,
        max_value=st.session_state.parametri['anni_totali'],
        value=st.session_state.parametri.get('inizio_pensione_anni', 30),
        step=1,
        help="Tra quanti anni inizierai a ricevere la pensione pubblica."
    )

def run_simulation_wrapper():
    """
    Raccoglie tutti i parametri dalla sidebar, li valida (se necessario)
    e lancia la simulazione completa.
    """
    p = st.session_state.get('parametri', {})
    
    # Validazione del portfolio prima di procedere
    if not np.isclose(st.session_state.get('portfolio', {}).get("Allocazione (%)", 0).sum(), 100):
        st.sidebar.error("L'allocazione del portafoglio deve essere esattamente 100% per eseguire la simulazione.")
        st.stop()
        
    # Calcolo dei parametri derivati dal portfolio
    rendimento_medio_portfolio, volatilita_portfolio, ter_etf_portfolio = get_portfolio_summary()

    # Aggiorna il dizionario dei parametri con tutti i valori correnti
    st.session_state.parametri.update({
        'rendimento_medio': rendimento_medio_portfolio,
        'volatilita': volatilita_portfolio,
        'ter_etf': ter_etf_portfolio
        # Gli altri parametri sono già aggiornati nel session_state tramite i loro widget
    })

    with st.spinner('Simulazione in corso... Questo potrebbe richiedere qualche istante.'):
        try:
            # Pulisce i risultati precedenti prima di una nuova simulazione
            if 'risultati' in st.session_state:
                del st.session_state.risultati
            
            # Esegui la simulazione.
            st.session_state.risultati = engine.run_full_simulation(st.session_state.parametri)
            
            st.success('Simulazione completata con successo!')
            st.query_params.clear()
        except ValueError as e:
            st.error(f"Errore nei parametri: {e}")
        except Exception as e:
            st.error(f"Si è verificato un errore inaspettato durante la simulazione: {e}")

# Pulsante per eseguire la simulazione
if st.sidebar.button('🚀 Esegui Simulazione', type="primary", use_container_width=True):
    run_simulation_wrapper()

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

    st.header("Riepilogo Entrate in Pensione (Valori Reali Medi)")
    stats_prelievi = st.session_state.risultati['statistiche_prelievi']
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Prelievo Medio dal Patrimonio", f"€ {stats_prelievi['prelievo_reale_medio']:,.0f}")
    col2.metric("Pensione Pubblica Annua", f"€ {stats_prelievi['pensione_pubblica_reale_annua']:,.0f}")
    col3.metric("Rendita Media da Fondo Pensione", f"€ {stats_prelievi['rendita_fp_reale_media']:,.0f}")
    col4.metric("TOTALE ENTRATE MEDIE ANNUE", f"€ {stats_prelievi['totale_reale_medio_annuo']:,.0f}")

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
        
        st.subheader("Come potrebbe evolvere il mio patrimonio? (Visione d'insieme)")
        st.markdown("""
        Questo primo grafico ti dà una visione d'insieme, un **"cono di probabilità"**. Non mostra una singola previsione, ma l'intera gamma di risultati possibili, tenendo conto dell'incertezza dei mercati.
        - **La linea rossa (Mediana):** È lo scenario più probabile (50° percentile). Metà delle simulazioni hanno avuto un risultato migliore, metà peggiore.
        - **Le aree colorate:** Rappresentano gli intervalli di confidenza. L'area più scura (25°-75° percentile) è dove il tuo patrimonio ha una buona probabilità di trovarsi. L'area più chiara (10°-90°) mostra gli scenari più estremi, sia positivi che negativi.
        """)
        fig_reale = plot_percentile_chart(
            dati_grafici['reale'], 'Evoluzione Patrimonio Reale (Tutti gli Scenari)', 'Patrimonio Reale (€)', 
            color_median='#C00000', color_fill='#C00000',
            anni_totali=params['anni_totali'],
            eta_iniziale=params['eta_iniziale']
        )
        fig_reale.add_vline(x=params['eta_iniziale'] + params['anni_inizio_prelievo'], line_width=2, line_dash="dash", line_color="grey", annotation_text="Inizio Prelievi")
        st.plotly_chart(fig_reale, use_container_width=True)
        st.markdown("<div style='text-align: center; font-size: 0.9em; font-style: italic;'>Questo è il grafico della verità. Tiene conto dell'inflazione, mostrando il vero potere d'acquisto.</div>", unsafe_allow_html=True)
        st.markdown("---")

        st.subheader("Quali sono i percorsi possibili? (Visione di dettaglio)")
        st.markdown("""
        Se il grafico precedente era una "mappa meteorologica", questo è come guardare le traiettorie di 50 aerei diversi che volano nella stessa tempesta. Ogni linea colorata è **una delle possibili vite del tuo portafoglio** tra le migliaia simulate.
        
        Questo grafico ti aiuta a capire la natura caotica dei mercati: alcuni percorsi sono fortunati (linee che finiscono in alto), altri meno. La **linea rossa più spessa** è sempre lo scenario mediano, il tuo punto di riferimento. Osserva come, nonostante le partenze simili, le traiettorie divergano enormemente nel tempo. Questo è il motivo per cui diversificare e avere un piano a lungo termine è fondamentale.
        """)
        fig_spaghetti = plot_spaghetti_chart(
            dati_grafici['reale'], 'Traiettorie Individuali del Patrimonio Reale', 'Patrimonio Reale (€)',
            '#FF0000', params['anni_totali'], params['anni_inizio_prelievo'],
            eta_iniziale=params['eta_iniziale']
        )
        st.plotly_chart(fig_spaghetti, use_container_width=True)
        st.markdown("---")

        st.subheader("E in termini nominali, senza inflazione?")
        fig_nominale = plot_percentile_chart(
            dati_grafici['nominale'], 'Evoluzione Patrimonio Nominale (Tutti gli Scenari)', 'Patrimonio (€)',
            color_median='#4472C4', color_fill='#4472C4',
            anni_totali=params['anni_totali'],
            eta_iniziale=params['eta_iniziale']
        )
        fig_nominale.add_vline(x=params['eta_iniziale'] + params['anni_inizio_prelievo'], line_width=2, line_dash="dash", line_color="grey", annotation_text="Inizio Prelievi")
        st.plotly_chart(fig_nominale, use_container_width=True)
        st.markdown("<div style='text-align: center; font-size: 0.9em; font-style: italic;'>Questo grafico mostra il valore 'nominale', cioè quanti Euro vedrai scritti sul tuo estratto conto in futuro, senza considerare l'inflazione.</div>", unsafe_allow_html=True)

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
            ('Fonte: Conto Corrente', 'prelievi_da_banca_nominali'),
            ('Fonte: Vendita ETF', 'prelievi_da_etf_nominali'),
            ('Vendita ETF (Rebalance)', 'vendite_rebalance_nominali'),
            ('Prelievo Effettivo (Reale)', 'prelievi_effettivi_reali'),
            ('Pensione Pubblica (Nom.)', 'pensioni_pubbliche_nominali'),
            ('Rendita FP (Nom.)', 'rendite_fp_nominali'),
            ('Liquidazione FP (Nom.)', 'fp_liquidato_nominale'),
            # Per i saldi, partiamo dall'anno 1 per allinearli con gli anni del dataframe
            ('Patrimonio Banca (Nom.)', 'saldo_banca_nominale'),
            ('Patrimonio ETF (Nom.)', 'saldo_etf_nominale'),
            ('Patrimonio FP (Nom.)', 'saldo_fp_nominale')
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
        }))

        with st.expander("Guida alla Lettura della Tabella"):
            st.markdown("""
            - **Obiettivo Prelievo vs Prelievo Effettivo**: L''Obiettivo' è quanto vorresti prelevare. L''Effettivo' è quanto prelevi realmente. Se hai pochi soldi, l''Effettivo' sarà più basso.
            - **Fonte Conto vs Fonte ETF**: Mostrano da dove provengono i soldi per il prelievo. Prima si usa la liquidità sul conto, poi si vendono gli ETF.
            - **Vendita ETF (Rebalance)**: NON sono soldi spesi. Sono vendite fatte per ridurre il rischio (seguendo il Glidepath). I soldi vengono spostati da ETF a liquidità.
            - **Liquidazione Capitale FP**: Somma che ricevi tutta in una volta dal fondo pensione all'età scelta. Aumenta di molto la tua liquidità in quell'anno.
            - **Entrate Anno (Reali)**: La somma di tutte le tue entrate (prelievi, pensioni) in potere d'acquisto di oggi. Questa cifra misura il tuo vero tenore di vita annuale.
            """) 