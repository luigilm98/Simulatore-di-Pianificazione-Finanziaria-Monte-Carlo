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

def plot_percentile_chart(data, title, y_title, color_median, color_fill, anni_totali, anni_inizio_prelievo=None):
    """Crea un grafico a 'cono' con i percentili."""
    fig = go.Figure()
    anni_asse_x = np.linspace(0, anni_totali, data.shape[1])

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
        hovertemplate='Anno %{x:.1f}<br>Valore Mediano: €%{y:,.0f}<extra></extra>'
    ))

    if anni_inizio_prelievo is not None:
        fig.add_vline(x=anni_inizio_prelievo, line_width=2, line_dash="dash", line_color="white",
                      annotation_text="Inizio Prelievi", annotation_position="top left")
    
    fig.update_layout(
        title=title,
        xaxis_title="Anni",
        yaxis_title=y_title,
        yaxis_tickformat="€,d",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        template="plotly_dark"
    )
    return fig

def plot_histogram(data, anni_totali):
    fig = go.Figure(data=[go.Histogram(x=data, nbinsx=30, marker_color='#4472C4')])
    fig.update_layout(
        title_text='Distribuzione del Patrimonio Finale Reale',
        xaxis_title_text='Patrimonio Finale (Potere d\'Acquisto Odierno)',
        yaxis_title_text='Numero di Simulazioni',
        bargap=0.1,
        template="plotly_dark"
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
        yaxis_range=[0, 1.01],
        template="plotly_dark"
    )
    return fig

def plot_income_composition(data, anni_totali):
    anni_asse_x_annuale = np.arange(anni_totali + 1)
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
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        template="plotly_dark"
    )
    return fig

def plot_asset_allocation(data, anni_totali):
    anni_asse_x_annuale = np.arange(anni_totali + 1)
    banca_reale = data['saldo_banca_reale']
    etf_reale = data['saldo_etf_reale']
    fp_reale = data.get('saldo_fp_reale', np.zeros_like(banca_reale)) # Compatibilità
    
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
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        template="plotly_dark"
    )
    return fig

def plot_income_cone_chart(data, anni_totali, anni_inizio_prelievo):
    """Crea un grafico a 'cono' per il reddito reale annuo."""
    fig = go.Figure()
    start_index = int(anni_inizio_prelievo)
    
    # La simulazione ha N+1 punti dati annuali (da anno 0 a anno N)
    end_index = int(anni_totali) + 1

    if start_index >= data.shape[1]:
        return go.Figure().update_layout(title="Periodo di decumulo non sufficiente per il grafico.", template="plotly_dark")

    anni_asse_x = np.arange(start_index, end_index)
    data_decumulo = data[:, start_index:end_index]

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
        hovertemplate='Età %{x:.0f}<br>Reddito Mediano: €%{y:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Quale Sarà il Mio Tenore di Vita? (Reddito Annuo Reale)",
        xaxis_title="Età",
        yaxis_title="Reddito Annuo Reale (€ di oggi)",
        yaxis_tickformat="€,d",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        template="plotly_dark"
    )
    return fig

def plot_worst_scenarios_chart(data, patrimoni_finali, anni_totali):
    """Mostra le traiettorie dei peggiori scenari."""
    fig = go.Figure()
    anni_asse_x = np.linspace(0, anni_totali, data.shape[1])
    
    for i, traiettoria in enumerate(data):
        fig.add_trace(go.Scatter(
            x=anni_asse_x, 
            y=traiettoria, 
            mode='lines',
            line=dict(width=1.5, color='rgba(255, 127, 14, 0.7)'),
            name=f"Scenario {i+1} (Finale: €{patrimoni_finali[i]:,.0f})",
            hovertemplate='Anno %{x:.1f}<br>Patrimonio: €%{y:,.0f}<extra></extra>'
        ))

    fig.update_layout(
        title='Traiettorie dei Peggiori Scenari (5%)',
        xaxis_title='Anni',
        yaxis_title='Patrimonio Reale (€ di oggi)',
        yaxis_tickformat="€,d",
        template="plotly_dark",
        showlegend=False
    )
    return fig

def get_default_portfolio():
    """Restituisce la configurazione di default del portafoglio."""
    return {
        "nome_simulazione": "La Mia Pensione",
        "eta_iniziale": 27, "capitale_iniziale": 50000, "etf_iniziale": 100000,
        "contributo_mensile_banca": 100, "contributo_mensile_etf": 1500,
        "rendimento_medio": 0.075, "volatilita": 0.15, "inflazione": 0.02,
        "anni_totali": 80,
        "strategia_prelievo": "REGOLA_4_PERCENTO",
        "prelievo_annuo": 12000, "percentuale_regola_4": 0.04,
        "banda_guardrail": 0.10,
        "anni_inizio_prelievo": 35,
        "tassazione_capital_gain": 0.26, "ter_etf": 0.0020, "costo_fisso_etf_mensile": 0,
        "attiva_glidepath": False, "inizio_glidepath_anni": 20, "fine_glidepath_anni": 40, "allocazione_etf_finale": 0.333,
        "attiva_fondo_pensione": False, "contributo_annuo_fp": 5000, "rendimento_medio_fp": 0.045, "volatilita_fp": 0.08, "ter_fp": 0.01,
        "eta_ritiro_fp": 67, "tipo_liquidazione_fp": "Capitale", "aliquota_finale_fp": 0.15,
        "inizio_pensione_anni": 40, "pensione_pubblica_annua": 20000,
        "n_simulazioni": 1000
    }

def run_simulation(params):
    """Esegue la simulazione e salva i risultati nello stato della sessione."""
    try:
        prelievo_annuo_da_usare = params['prelievo_annuo']
        
        with st.spinner("Calcolo in corso... Questo potrebbe richiedere alcuni secondi."):
            results = engine.esegui_simulazioni(params, prelievo_annuo_da_usare)
            st.session_state.simulation_results = results
            st.session_state.simulation_params = params
            st.session_state.show_details = False
            st.session_state.show_advanced_charts = False
    except ValueError as e:
        st.error(f"Errore nei parametri: {e}")
        st.session_state.simulation_results = None

# --- Inizio Interfaccia Streamlit ---
st.set_page_config(layout="wide", page_title="Simulatore Finanziario Monte Carlo")

# --- CSS Custom ---
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
    }
    .stMetric {
        border: 1px solid #444;
        border-radius: 8px;
        padding: 10px;
        background-color: #2a2a2e;
    }
</style>
""", unsafe_allow_html=True)


# --- Barra Laterale (Sidebar) ---
with st.sidebar:
    st.image("https://www.copilot.com/logo.svg", width=150) # Placeholder
    st.title("Configurazione Simulazione")

    simulations_list = load_simulations()
    
    st.header("Gestione Scenari")
    
    if 'simulation_params' not in st.session_state:
        st.session_state.simulation_params = get_default_portfolio()

    selected_sim = st.selectbox("Carica Scenario Salvato", options=[f"{s['name']} ({s['timestamp']})" for s in simulations_list], index=None, placeholder="Seleziona uno scenario...")
    
    if selected_sim:
        sim_data = next(s for s in simulations_list if f"{s['name']} ({s['timestamp']})" == selected_sim)
        if st.button("Carica Scenario Selezionato"):
            loaded_data = load_simulation_data(sim_data['filename'])
            st.session_state.simulation_params = loaded_data['parameters']
            st.session_state.simulation_results = loaded_data['results']
            st.rerun()

    p = st.session_state.simulation_params
    
    p['nome_simulazione'] = st.text_input("Nome dello Scenario", value=p['nome_simulazione'])

    with st.expander("👤 Profilo Anagrafico e Capitale", expanded=True):
        p['eta_iniziale'] = st.number_input("Età Iniziale", min_value=18, max_value=100, value=p['eta_iniziale'])
        p['capitale_iniziale'] = st.number_input("Liquidità Iniziale (€)", min_value=0, value=p['capitale_iniziale'], step=1000)
        p['etf_iniziale'] = st.number_input("Patrimonio ETF Iniziale (€)", min_value=0, value=p['etf_iniziale'], step=1000)
        p['contributo_mensile_banca'] = st.number_input("Risparmio Mensile in Liquidità (€)", min_value=0, value=p['contributo_mensile_banca'], step=50)
        p['contributo_mensile_etf'] = st.number_input("Investimento Mensile in ETF (€)", min_value=0, value=p['contributo_mensile_etf'], step=50)
        p['anni_totali'] = st.number_input("Orizzonte Temporale (Anni)", min_value=1, max_value=100, value=p['anni_totali'])

    with st.expander("📈 Ipotesi di Mercato", expanded=True):
        p['rendimento_medio'] = st.slider("Rendimento Medio Annuo Lordo ETF (%)", 0.0, 20.0, p['rendimento_medio'] * 100, 0.5) / 100
        p['volatilita'] = st.slider("Volatilità Annua ETF (%)", 0.0, 40.0, p['volatilita'] * 100, 1.0) / 100
        p['inflazione'] = st.slider("Inflazione Media Annua (%)", 0.0, 10.0, p['inflazione'] * 100, 0.25) / 100

    with st.expander("💰 Strategia di Prelievo", expanded=False):
        p['anni_inizio_prelievo'] = st.number_input("Inizio Prelievo tra (Anni)", min_value=0, max_value=p['anni_totali'], value=p['anni_inizio_prelievo'])
        p['strategia_prelievo'] = st.selectbox("Strategia di Prelievo", 
            options=['FISSO', 'REGOLA_4_PERCENTO', 'GUARDRAIL'], 
            index=['FISSO', 'REGOLA_4_PERCENTO', 'GUARDRAIL'].index(p.get('strategia_prelievo', 'REGOLA_4_PERCENTO')),
            help="""
            - **FISSO**: Prelievo di un importo fisso, rivalutato per l'inflazione.
            - **REGOLA_4%**: Prelievo del 4% del patrimonio a inizio anno.
            - **GUARDRAIL**: Come FISSO, ma con aggiustamenti del 10% se il tasso di prelievo supera certe soglie.
            """)
        
        if p['strategia_prelievo'] == 'FISSO':
            p['prelievo_annuo'] = st.number_input("Importo Prelievo Fisso Annuo (€)", min_value=0, value=p['prelievo_annuo'], step=1000)
        elif p['strategia_prelievo'] == 'REGOLA_4_PERCENTO':
            p['percentuale_regola_4'] = st.slider("Percentuale Regola 4% / Prelievo Iniziale (%)", 0.0, 10.0, p.get('percentuale_regola_4', 0.04) * 100, 0.1) / 100
        elif p['strategia_prelievo'] == 'GUARDRAIL':
            st.number_input("Importo Prelievo Fisso Annuo (€)", min_value=0, value=p['prelievo_annuo'], step=1000, key='prelievo_guardrail') # Usa una chiave diversa
            p['percentuale_regola_4'] = st.slider("Tasso di Prelievo di Riferimento (%)", 0.0, 10.0, p.get('percentuale_regola_4', 0.04) * 100, 0.1) / 100
            p['banda_guardrail'] = st.slider("Banda Guardrail (%)", 0.0, 50.0, p.get('banda_guardrail', 0.20) * 100, 1.0) / 100

    with st.expander("📉 Asset Allocation Dinamica (Glidepath)", expanded=False):
        p['attiva_glidepath'] = st.checkbox("Attiva Glidepath", value=p.get('attiva_glidepath', False))
        if p['attiva_glidepath']:
            p['inizio_glidepath_anni'] = st.number_input("Inizio Glidepath tra (Anni)", 0, p['anni_totali'], p.get('inizio_glidepath_anni', 20))
            p['fine_glidepath_anni'] = st.number_input("Fine Glidepath tra (Anni)", p['inizio_glidepath_anni'], p['anni_totali'], p.get('fine_glidepath_anni', 40))
            p['allocazione_etf_finale'] = st.slider("Allocazione ETF Finale (%)", 0.0, 100.0, p.get('allocazione_etf_finale', 0.333) * 100, 1.0) / 100

    with st.expander("🇮🇹 Tassazione e Costi (Italia)", expanded=False):
        p['tassazione_capital_gain'] = st.slider("Tassazione Capital Gain (%)", 0.0, 40.0, p['tassazione_capital_gain'] * 100, 0.5) / 100
        p['ter_etf'] = st.slider("Costo Annuo ETF (TER) (%)", 0.0, 2.0, p['ter_etf'] * 100, 0.01) / 100
        p['costo_fisso_etf_mensile'] = st.number_input("Costi Fissi Mensili Conto Titoli (€)", 0.0, 100.0, p.get('costo_fisso_etf_mensile', 0.0), 0.5)

    with st.expander("🏦 Fondo Pensione", expanded=False):
        p['attiva_fondo_pensione'] = st.checkbox("Attiva Fondo Pensione", value=p['attiva_fondo_pensione'])
        if p['attiva_fondo_pensione']:
            p['contributo_annuo_fp'] = st.number_input("Contributo Annuo Fondo Pensione (€)", 0, 10000, p['contributo_annuo_fp'], 100)
            p['rendimento_medio_fp'] = st.slider("Rendimento Medio Annuo FP (%)", 0.0, 15.0, p['rendimento_medio_fp'] * 100, 0.25) / 100
            p['volatilita_fp'] = st.slider("Volatilità Annua FP (%)", 0.0, 25.0, p.get('volatilita_fp', 0.08) * 100, 0.5) / 100
            p['ter_fp'] = st.slider("Costo Annuo FP (TER) (%)", 0.0, 5.0, p['ter_fp'] * 100, 0.1) / 100
            p['eta_ritiro_fp'] = st.number_input("Età Ritiro Fondo Pensione", 57, 75, p['eta_ritiro_fp'])
            p['tipo_liquidazione_fp'] = st.selectbox("Tipo Liquidazione FP", ['Capitale', 'Rendita'], index=['Capitale', 'Rendita'].index(p['tipo_liquidazione_fp']))
            p['aliquota_finale_fp'] = st.slider("Aliquota Fiscale Finale FP (%)", 9.0, 23.0, p['aliquota_finale_fp'] * 100, 0.5) / 100
            
    with st.expander("💶 Altre Entrate", expanded=False):
        p['inizio_pensione_anni'] = st.number_input("Inizio Pensione Pubblica tra (Anni)", 0, p['anni_totali'], p['inizio_pensione_anni'])
        p['pensione_pubblica_annua'] = st.number_input("Importo Annuo Pensione Pubblica (€)", 0, 100000, p['pensione_pubblica_annua'], 1000)

    st.header("⚙️ Impostazioni Tecniche")
    p['n_simulazioni'] = st.select_slider("Numero di Simulazioni", options=[100, 500, 1000, 2000, 5000], value=p['n_simulazioni'])

    if st.button("Esegui Simulazione", type="primary", use_container_width=True):
        run_simulation(p)

# --- Pagina Principale ---
if 'simulation_results' not in st.session_state or st.session_state.simulation_results is None:
    st.title("Simulatore di Pianificazione Finanziaria Monte Carlo")
    st.markdown("### Benvenuto! Configura i parametri della tua simulazione nella barra laterale e clicca su 'Esegui Simulazione' per iniziare.")
    st.info("Questo strumento ti aiuta a visualizzare migliaia di possibili futuri finanziari per prendere decisioni più consapevoli.")

else:
    # Carica risultati e parametri dallo stato della sessione
    results = st.session_state.simulation_results
    params = st.session_state.simulation_params
    stats = results['statistiche']
    stat_patrimonio = stats['percentili_patrimonio_reale']
    stat_prelievi = stats.get('statistiche_prelievi', {k: 0 for k in ['mediana', 'p10', 'p90']})

    st.title(f"Risultati per: *{params['nome_simulazione']}*")
    st.markdown(f"*{params['n_simulazioni']} scenari calcolati in un orizzonte di {params['anni_totali']} anni.*")

    # --- Sezione Metriche Chiave ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="📉 Probabilità di Fallimento",
            value=f"{stats['tasso_fallimento']:.1%}",
            help="La percentuale di simulazioni in cui il patrimonio si è esaurito prima della fine dell'orizzonte temporale."
        )
    with col2:
        st.metric(
            label="Median Wealth at End",
            value=f"€{stat_patrimonio.get('mediana', 0):,.0f}",
            help="Il patrimonio finale reale (potere d'acquisto odierno) nello scenario mediano (50° percentile)."
        )
    with col3:
        reddito_mediano = stat_prelievi.get('mediana', 0)
        st.metric(
            label="Reddito Annuo Reale Mediano",
            value=f"€{reddito_mediano:,.0f}",
            help="Il reddito annuo reale mediano (prelievi + pensioni) che puoi sostenere durante il decumulo. Calcolato solo sugli scenari di successo."
        )
    with col4:
        patrimonio_finale_in_anni_spesa = stat_patrimonio.get('mediana', 0) / reddito_mediano if reddito_mediano > 0 else 0
        st.metric(
            label="Patrimonio Finale in Anni di Spesa",
            value=f"{patrimonio_finale_in_anni_spesa:.1f} Anni",
            help="Quanti anni di spesa (basati sul reddito mediano) rappresenta il tuo patrimonio finale mediano. Utile per capire il 'margine di sicurezza'."
        )

    # --- Tabbed Interface per i Grafici ---
    tab1, tab2, tab3 = st.tabs(["📊 Grafici Principali", "📈 Grafici Avanzati", "📄 Dettaglio Flussi di Cassa"])
    
    with tab1:
        st.header("Evoluzione del Patrimonio e del Tenore di Vita")
        
        # Grafico Evoluzione Patrimonio
        patrimoni_reali = np.array(results['patrimoni_reali'])
        fig_patrimonio = plot_percentile_chart(
            patrimoni_reali,
            "Evoluzione Patrimonio Reale (Tutti gli Scenari)",
            "Patrimonio Reale (€)",
            '#FFC000', '#FFC000',
            params['anni_totali'],
            params['anni_inizio_prelievo']
        )
        st.plotly_chart(fig_patrimonio, use_container_width=True)

        # Grafico Tenore di Vita
        reddito_reale = np.array(results['reddito_totale_reale'])
        eta_x_axis = np.arange(params['anni_totali'] + 1) + params['eta_iniziale']
        fig_reddito = plot_income_cone_chart(reddito_reale, params['anni_totali'], params['anni_inizio_prelievo'])
        fig_reddito.update_xaxes(tickvals=eta_x_axis[::5], ticktext=[str(x) for x in eta_x_axis[::5]])
        st.plotly_chart(fig_reddito, use_container_width=True)

        st.info("""
        **Come leggere questi grafici?**
        - **Linea Azzurra/Gialla**: Rappresenta lo scenario mediano (50° percentile), il risultato più probabile.
        - **Aree Colorate**: Mostrano l'incertezza. L'area più scura (25-75 percentile) contiene il 50% degli scenari più comuni. L'area più chiara (10-90 percentile) ne contiene l'80%. Più ampie sono le aree, maggiore è l'incertezza del futuro.
        """)

    with tab2:
        st.header("Analisi Avanzate dello Scenario")
        dati_grafici_avanzati = results.get("dati_grafici_avanzati", {})
        
        col1, col2 = st.columns(2)
        with col1:
            if "dati_mediana" in dati_grafici_avanzati:
                fig_composizione_reddito = plot_income_composition(dati_grafici_avanzati['dati_mediana'], params['anni_totali'])
                st.plotly_chart(fig_composizione_reddito, use_container_width=True)

            if "prob_successo_nel_tempo" in dati_grafici_avanzati:
                fig_prob_successo = plot_success_probability(dati_grafici_avanzati['prob_successo_nel_tempo'], params['anni_totali'])
                st.plotly_chart(fig_prob_successo, use_container_width=True)
                
        with col2:
            if "dati_mediana" in dati_grafici_avanzati:
                fig_allocazione = plot_asset_allocation(dati_grafici_avanzati['dati_mediana'], params['anni_totali'])
                st.plotly_chart(fig_allocazione, use_container_width=True)

            if "worst_scenarios" in dati_grafici_avanzati:
                worst_data = dati_grafici_avanzati['worst_scenarios']
                fig_worst = plot_worst_scenarios_chart(
                    np.array(worst_data['traiettorie']),
                    np.array(worst_data['patrimoni_finali']),
                    params['anni_totali']
                )
                st.plotly_chart(fig_worst, use_container_width=True)


    with tab3:
        st.header("Dettaglio Flussi di Cassa (Scenario Mediano)")
        st.write("Questa tabella mostra i flussi di cassa annuali per lo scenario mediano (50° percentile). Tutti i valori sono in termini reali (potere d'acquisto odierno).")

        dati_mediana = dati_grafici_avanzati.get("dati_mediana", {})
        
        # Prepara i dati per la tabella, gestendo chiavi mancanti per vecchie simulazioni
        anni = np.arange(params['anni_totali'] + 1)
        eta = anni + params['eta_iniziale']
        
        df_data = {
            "Età": eta,
            "Patrimonio Iniziale Reale": np.median(patrimoni_reali[:, ::12], axis=0),
            "Reddito Reale Annuo": np.median(results.get('reddito_totale_reale', np.zeros_like(patrimoni_reali)), axis=0),
            "Prelievi Effettivi Reali": dati_mediana.get('prelievi_effettivi_reali', np.zeros(anni.size)),
            "Pensione Pubblica Reale": dati_mediana.get('pensioni_pubbliche_reali', np.zeros(anni.size)),
            "Rendita FP Reale": dati_mediana.get('rendite_fp_reali', np.zeros(anni.size)),
            "Liquidazione FP Nominale": dati_mediana.get('fp_liquidato_nominale', np.zeros(anni.size)),
            "Vendite per Rebalance": dati_mediana.get('vendite_rebalance_nominali', np.zeros(anni.size)),
            "Tasse su Rebalance": np.median(results.get('tasse_rebalance_nominali', np.zeros_like(patrimoni_reali)), axis=0)
        }
        
        df = pd.DataFrame(df_data)
        
        # Formattazione per la visualizzazione
        styled_df = df.style.format({
            "Patrimonio Iniziale Reale": "€{:,.0f}",
            "Reddito Reale Annuo": "€{:,.0f}",
            "Prelievi Effettivi Reali": "€{:,.0f}",
            "Pensione Pubblica Reale": "€{:,.0f}",
            "Rendita FP Reale": "€{:,.0f}",
            "Liquidazione FP Nominale": "€{:,.0f}",
            "Vendite per Rebalance": "€{:,.0f}",
            "Tasse su Rebalance": "€{:,.0f}"
        }).set_properties(**{'text-align': 'right'}).hide(axis="index")
        
        st.dataframe(styled_df, use_container_width=True, height=500)


    st.header("Gestione Scenario Corrente")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Salva Scenario Corrente", use_container_width=True):
            save_simulation(params['nome_simulazione'], params, results)
    with col2:
        # Aggiungi un bottone per eliminare lo scenario caricato se applicabile
        pass # Logica di eliminazione gestita nella sidebar 