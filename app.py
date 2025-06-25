import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from data_loader import DataLoader
import os
import json

st.set_page_config(page_title="Simulatore di Resilienza Finanziaria 2.0", layout="wide")
st.title("Simulatore di Resilienza Finanziaria Personale (v2.0)")

st.markdown("""
Questa versione è progettata per offrire analisi realistiche, stress test, raccomandazioni e report professionali per la pianificazione finanziaria personale.

**Carica il tuo profilo, scegli gli scenari di stress, e ricevi un report dettagliato sulla resilienza del tuo piano!**
""")

# --- 1. Caricamento profilo utente ---
st.header("1. Carica il tuo profilo finanziario")
with st.form("profilo_utente"):
    col1, col2, col3 = st.columns(3)
    with col1:
        eta = st.number_input("Età attuale", min_value=18, max_value=100, value=40)
        orizzonte = st.number_input("Orizzonte temporale (anni)", min_value=5, max_value=70, value=40)
    with col2:
        capitale_iniziale = st.number_input("Capitale iniziale (€)", min_value=0, value=100000, step=1000)
        reddito_annuo = st.number_input("Reddito annuo netto (€)", min_value=0, value=35000, step=1000)
    with col3:
        spese_annue = st.number_input("Spese annue (€)", min_value=0, value=25000, step=1000)
        obiettivo_prelievo = st.number_input("Obiettivo prelievo annuo in pensione (€)", min_value=0, value=20000, step=1000)

    st.markdown("**Asset Allocation (%)**")
    col4, col5, col6 = st.columns(3)
    with col4:
        pct_azioni = st.slider("Azioni", 0, 100, 60, 5)
    with col5:
        pct_obblig = st.slider("Obbligazioni", 0, 100, 30, 5)
    with col6:
        pct_liquidita = st.slider("Liquidità", 0, 100, 10, 5)
    if pct_azioni + pct_obblig + pct_liquidita != 100:
        st.error("La somma delle percentuali deve essere 100%.")
    submitted = st.form_submit_button("Salva profilo")

# Salvataggio profilo utente fuori dal form
if submitted and pct_azioni + pct_obblig + pct_liquidita == 100:
    st.session_state['profilo_utente'] = {
        'eta': eta,
        'orizzonte': orizzonte,
        'capitale_iniziale': capitale_iniziale,
        'reddito_annuo': reddito_annuo,
        'spese_annue': spese_annue,
        'obiettivo_prelievo': obiettivo_prelievo,
        'pct_azioni': pct_azioni,
        'pct_obblig': pct_obblig,
        'pct_liquidita': pct_liquidita
    }
    st.success("Profilo utente salvato! Ora puoi scegliere gli scenari di stress.")

# --- 2. Scelta scenari di stress ---
st.header("2. Scegli scenari di stress")
with st.form("scenari_stress"):
    st.markdown("**Crisi di Mercato**")
    crash = st.checkbox("Simula un crash di mercato (es. -50% al 10° anno)", value=True)
    crash_severity = st.slider("Entità crash (%)", -90, -10, -50, 5)
    crash_year = st.slider("Anno del crash", 1, 40, 10)

    st.markdown("**Shock Inflattivo**")
    infl_shock = st.checkbox("Simula shock inflattivo (es. inflazione 10% per 5 anni)", value=False)
    infl_rate = st.slider("Tasso inflazione shock (%)", 2, 20, 10)
    infl_duration = st.slider("Durata shock (anni)", 1, 10, 5)

    st.markdown("**Longevità Estrema**")
    longevity = st.checkbox("Simula longevità superiore alla media (+10 anni)", value=False)
    extra_years = st.slider("Anni extra di vita", 0, 30, 10)

    st.markdown("**Spese Imprevisti**")
    spesa_shock = st.checkbox("Simula spesa imprevista (es. 50.000€ al 20° anno)", value=False)
    spesa_importo = st.number_input("Importo spesa imprevista (€)", min_value=0, value=50000, step=1000)
    spesa_anno = st.slider("Anno spesa imprevista", 1, 40, 20)

    submitted_stress = st.form_submit_button("Salva scenari di stress")

# Salvataggio scenari di stress fuori dal form
if submitted_stress:
    st.session_state['scenari_stress'] = {
        'crash': crash,
        'crash_severity': crash_severity,
        'crash_year': crash_year,
        'infl_shock': infl_shock,
        'infl_rate': infl_rate,
        'infl_duration': infl_duration,
        'longevity': longevity,
        'extra_years': extra_years,
        'spesa_shock': spesa_shock,
        'spesa_importo': spesa_importo,
        'spesa_anno': spesa_anno
    }
    st.success("Scenari di stress salvati! Ora puoi eseguire la simulazione.")

# --- 3. Esecuzione simulazione e visualizzazione risultati ---
st.header("3. Esegui simulazione e visualizza risultati")
if 'profilo_utente' not in st.session_state or 'scenari_stress' not in st.session_state:
    st.info("⚠️ Inserisci prima il profilo utente e gli scenari di stress per poter eseguire la simulazione.")
else:
    if st.button("Esegui simulazione", type="primary"):
        loader = DataLoader()
        df_hist = loader.load_sample_historical_data(years=st.session_state['profilo_utente']['orizzonte'])
        patrimonio = [st.session_state['profilo_utente']['capitale_iniziale']]
        for i, row in df_hist.iterrows():
            last = patrimonio[-1]
            # Calcolo rendimento composito (azioni, obbligazioni, liquidità)
            r_az = (st.session_state['profilo_utente']['pct_azioni']/100) * row['equity_return']
            r_ob = (st.session_state['profilo_utente']['pct_obblig']/100) * 0.02  # 2% fisso obbligazioni
            r_liq = (st.session_state['profilo_utente']['pct_liquidita']/100) * 0.005  # 0.5% fisso liquidità
            rendimento = r_az + r_ob + r_liq
            infl = row['inflation']
            nuovo = last * (1 + rendimento)
            # Applica stress test crash
            if st.session_state['scenari_stress']['crash'] and i+1 == st.session_state['scenari_stress']['crash_year']:
                nuovo = nuovo * (1 + st.session_state['scenari_stress']['crash_severity']/100)
            # Applica shock inflattivo
            if st.session_state['scenari_stress']['infl_shock'] and st.session_state['scenari_stress']['infl_duration'] > 0:
                if st.session_state['scenari_stress']['crash_year'] <= i+1 < st.session_state['scenari_stress']['crash_year'] + st.session_state['scenari_stress']['infl_duration']:
                    infl = st.session_state['scenari_stress']['infl_rate']/100
            # Applica spesa imprevista
            if st.session_state['scenari_stress']['spesa_shock'] and i+1 == st.session_state['scenari_stress']['spesa_anno']:
                nuovo -= st.session_state['scenari_stress']['spesa_importo']
            # Aggiungi reddito e togli spese
            nuovo += st.session_state['profilo_utente']['reddito_annuo'] - st.session_state['profilo_utente']['spese_annue']
            patrimonio.append(nuovo)
        anni = np.arange(st.session_state['profilo_utente']['eta'], st.session_state['profilo_utente']['eta'] + st.session_state['profilo_utente']['orizzonte'] + 1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=anni, y=patrimonio, mode='lines+markers', name='Patrimonio simulato'))
        fig.update_layout(title='Evoluzione del Patrimonio (Rolling Window Storica con Stress Test)', xaxis_title='Età', yaxis_title='Patrimonio (€)')
        st.plotly_chart(fig, use_container_width=True)
        st.success("Simulazione completata! Questo è un esempio di rolling window con asset allocation e stress test.")

        # --- Analisi rischio e raccomandazioni ---
        patrimonio_arr = np.array(patrimonio)
        min_patrimonio = np.min(patrimonio_arr)
        drawdown = np.min(patrimonio_arr / np.maximum.accumulate(patrimonio_arr)) - 1
        impoverimento = np.any(patrimonio_arr < 0)
        fine = patrimonio_arr[-1]
        anni_sopravvivenza = len(patrimonio_arr) - 1
        rischio_longevita = fine < st.session_state['profilo_utente']['obiettivo_prelievo'] * 10  # Arbitrario: meno di 10 anni di prelievo

        # Salva risultati per confronto scenari PRIMA di qualsiasi output
        st.session_state['last_simulation'] = {
            'profilo_utente': st.session_state['profilo_utente'],
            'scenari_stress': st.session_state['scenari_stress'],
            'patrimonio': patrimonio,
            'anni': list(anni),
            'indicatori': {
                'patrimonio_finale': float(fine),
                'drawdown': float(drawdown),
                'impoverimento': bool(impoverimento),
                'rischio_longevita': bool(rischio_longevita)
            },
            'raccomandazioni': raccomandazioni
        }

        # Warning
        if impoverimento:
            st.error("⚠️ Attenzione: il patrimonio è andato in negativo almeno una volta. Rischio impoverimento reale!")
        if drawdown < -0.5:
            st.warning(f"⚠️ Drawdown massimo superiore al 50%: {drawdown:.1%}")
        if rischio_longevita:
            st.warning("⚠️ Il patrimonio finale non copre almeno 10 anni di prelievi: rischio longevità!")

        # Raccomandazioni
        st.markdown("---")
        st.subheader("Raccomandazioni automatiche")
        raccomandazioni = []
        if impoverimento:
            raccomandazioni.append("Riduci le spese annuali o aumenta il capitale iniziale per evitare il rischio di andare in rosso.")
        if drawdown < -0.5:
            raccomandazioni.append("Considera di ridurre la quota azionaria o aumentare la liquidità per ridurre la volatilità.")
        if rischio_longevita:
            raccomandazioni.append("Riduci il prelievo annuo o valuta una rendita vitalizia per coprire il rischio longevità.")
        if not raccomandazioni:
            raccomandazioni.append("Il piano appare resiliente: nessun rischio critico individuato con questi parametri.")
        for rac in raccomandazioni:
            st.info(rac)

        # --- Esportazione dati ---
        st.markdown("---")
        st.subheader("Esporta i dati della simulazione")
        df_export = pd.DataFrame({
            'Età': anni,
            'Patrimonio': patrimonio
        })
        st.download_button("Scarica dati in Excel", data=df_export.to_excel(index=False), file_name="simulazione.xlsx")
        st.download_button("Scarica dati in CSV", data=df_export.to_csv(index=False), file_name="simulazione.csv")

        # --- Generazione report PDF (base) ---
        from fpdf import FPDF
        import tempfile
        class PDF(FPDF):
            def header(self):
                self.set_font('Arial', 'B', 12)
                self.cell(0, 10, 'Report Simulazione Resilienza Finanziaria', 0, 1, 'C')
            def chapter_title(self, title):
                self.set_font('Arial', 'B', 11)
                self.cell(0, 8, title, 0, 1, 'L')
            def chapter_body(self, body):
                self.set_font('Arial', '', 10)
                self.multi_cell(0, 7, body)
        if st.button("Scarica report PDF"):
            pdf = PDF()
            pdf.add_page()
            pdf.chapter_title("Executive Summary")
            pdf.chapter_body(f"Patrimonio finale: €{fine:,.0f}\nDrawdown massimo: {drawdown:.1%}\nRischio impoverimento: {'SI' if impoverimento else 'NO'}\nRischio longevità: {'SI' if rischio_longevita else 'NO'}")
            pdf.chapter_title("Raccomandazioni")
            for rac in raccomandazioni:
                pdf.chapter_body(f"- {rac}")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
                pdf.output(tmpfile.name)
                tmpfile.seek(0)
                st.download_button("Scarica PDF", data=tmpfile.read(), file_name="report_simulazione.pdf")

# --- 4. Reportistica ---
st.header("4. Executive summary e limiti del modello")
st.info("""
**Limiti e assunzioni principali:**
- I rendimenti storici sono simulati e non garantiscono risultati futuri.
- Gli stress test sono semplificati e non coprono tutti i possibili eventi di vita reale.
- Il modello non tiene conto di fiscalità reale, costi sanitari, eredità, ecc.
- Le raccomandazioni sono indicative e non costituiscono consulenza finanziaria personalizzata.

**Usa questo strumento come supporto alle decisioni, non come unica fonte di verità.**
""")

# --- 5. Gestione e confronto scenari ---
st.header("5. Gestione e confronto scenari")
SCENARIO_DIR = "simulation_history"
os.makedirs(SCENARIO_DIR, exist_ok=True)

if 'last_simulation' in st.session_state:
    with st.form("save_scenario_form"):
        scenario_name = st.text_input("Nome scenario da salvare", "Scenario " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"))
        save_btn = st.form_submit_button("Salva scenario")

# Salvataggio scenario fuori dal form
if 'save_btn' in locals() and save_btn:
    fname = os.path.join(SCENARIO_DIR, scenario_name.replace(" ", "_") + ".json")
    with open(fname, "w") as f:
        json.dump(st.session_state['last_simulation'], f, indent=2)
    st.success(f"Scenario '{scenario_name}' salvato!")
    del st.session_state['last_simulation']

scenari_files = [f for f in os.listdir(SCENARIO_DIR) if f.endswith('.json')]
if scenari_files:
    st.markdown("**Scenari salvati:**")
    selected = st.multiselect("Seleziona scenari per il confronto", scenari_files)
    if selected:
        scenari = []
        for fname in selected:
            with open(os.path.join(SCENARIO_DIR, fname), "r") as f:
                scenari.append(json.load(f))
        # Grafico comparativo delle traiettorie patrimoniali
        fig = go.Figure()
        for i, sc in enumerate(scenari):
            fig.add_trace(go.Scatter(x=sc['anni'], y=sc['patrimonio'], mode='lines+markers', name=selected[i]))
        fig.update_layout(title="Confronto traiettorie patrimoniali", xaxis_title="Età", yaxis_title="Patrimonio (€)")
        st.plotly_chart(fig, use_container_width=True)
        # Tabella comparativa degli indicatori
        df_comp = pd.DataFrame([
            {
                'Scenario': selected[i],
                'Patrimonio finale': sc['indicatori']['patrimonio_finale'],
                'Drawdown': sc['indicatori']['drawdown'],
                'Impoverimento': sc['indicatori']['impoverimento'],
                'Rischio longevità': sc['indicatori']['rischio_longevita']
            } for i, sc in enumerate(scenari)
        ]).set_index('Scenario')
        st.dataframe(df_comp)
        # Esportazione tabella comparativa
        st.download_button("Scarica confronto in Excel", data=df_comp.to_excel(index=True), file_name="confronto_scenari.xlsx")
        st.download_button("Scarica confronto in CSV", data=df_comp.to_csv(index=True), file_name="confronto_scenari.csv")
        # Raccomandazioni a confronto
        st.markdown("---")
        st.subheader("Raccomandazioni per ciascun scenario")
        for i, sc in enumerate(scenari):
            st.markdown(f"**{selected[i]}**")
            for rac in sc.get('raccomandazioni', []):
                st.info(rac)
    # Pulsanti elimina
    for fname in scenari_files:
        if st.button(f"Elimina {fname}"):
            os.remove(os.path.join(SCENARIO_DIR, fname))
            st.experimental_rerun()
else:
    st.info("Nessuno scenario salvato ancora.")