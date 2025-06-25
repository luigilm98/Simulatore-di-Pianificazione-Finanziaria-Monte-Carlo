#!/usr/bin/env python3
"""
Script di debug per analizzare la logica di calcolo dei guadagni da investimento
"""
import numpy as np
import simulation_engine as engine
import matplotlib.pyplot as plt

def debug_single_simulation():
    """Esegue una singola simulazione con debug dettagliato"""
    
    # Parametri di test basati sui valori dello screenshot
    parametri_test = {
        'eta_iniziale': 27,
        'capitale_iniziale': 17000,
        'etf_iniziale': 600,
        'contributo_mensile_banca': 1300,
        'contributo_mensile_etf': 300,
        'anni_inizio_prelievo': 35,
        'prelievo_annuo': 12000,
        'n_simulazioni': 1,  # Solo una simulazione per debug
        'anni_totali': 80,
        'tassazione_capital_gain': 0.26,
        'ter_etf': 0.0022,
        'costo_fisso_etf_mensile': 0,
        'attiva_fondo_pensione': True,
        'rendimento_medio_fp': 0.04,
        'volatilita_fp': 0.08,
        'ter_fp': 0.01,
        'tassazione_rendimenti_fp': 0.20,
        'aliquota_finale_fp': 0.15,
        'eta_ritiro_fp': 67,
        'percentuale_capitale_fp': 0.5,
        'durata_rendita_fp_anni': 25,
        'strategia_ribilanciamento': 'GLIDEPATH',
        'inizio_glidepath_anni': 20,
        'fine_glidepath_anni': 40,
        'allocazione_etf_finale': 0.333,
        'allocazione_etf_fissa': 0.60,
        'imposta_bollo_titoli': 0.002,
        'imposta_bollo_conto': 34,
        'pensione_pubblica_annua': 8400,
        'inizio_pensione_anni': 40,
        'economic_model': "VOLATILE (CICLI BOOM-BUST)",
        'modalita_parametri_rendimento': "Combinazione Pesata",
        'peso_azioni': 0.6,
        'rendimento_medio': 0.06,
        'volatilita': 0.12,
        'inflazione': 0.025,
        'strategia_prelievo': 'FISSO',
        'percentuale_regola_4': 0.04,
        'banda_guardrail': 0.10,
        'indicizza_contributi_inflazione': True,
        'contributo_annuo_fp': 3000
    }
    
    print("=== DEBUG SIMULAZIONE SINGOLA ===")
    print(f"Capitale iniziale: €{parametri_test['capitale_iniziale']:,}")
    print(f"ETF iniziale: €{parametri_test['etf_iniziale']:,}")
    print(f"Contributo mensile banca: €{parametri_test['contributo_mensile_banca']:,}")
    print(f"Contributo mensile ETF: €{parametri_test['contributo_mensile_etf']:,}")
    print(f"Anni di accumulo: {parametri_test['anni_inizio_prelievo']}")
    print(f"Rendimento medio: {parametri_test['rendimento_medio']:.1%}")
    print(f"Volatilità: {parametri_test['volatilita']:.1%}")
    print()
    
    # Esegui la simulazione
    risultati = engine.run_full_simulation(parametri_test)
    
    # Analizza i risultati
    stats = risultati['statistiche']
    dati_mediana = risultati['dati_grafici_avanzati']['dati_mediana']
    
    print("=== RISULTATI ===")
    print(f"Patrimonio finale mediano (nominale): €{stats['patrimonio_finale_mediano_nominale']:,.0f}")
    print(f"Patrimonio finale mediano (reale): €{stats['patrimonio_finale_mediano_reale']:,.0f}")
    print(f"Contributi totali versati: €{stats['contributi_totali_versati_mediano_nominale']:,.0f}")
    print(f"Guadagni da investimento: €{stats['guadagni_accumulo_mediano_nominale']:,.0f}")
    print(f"Patrimonio all'inizio prelievi: €{stats['patrimonio_inizio_prelievi_mediano_nominale']:,.0f}")
    print()
    
    # Analisi dettagliata del patrimonio all'inizio prelievi
    idx_inizio_prelievo = parametri_test['anni_inizio_prelievo']
    patrimonio_banca_inizio = dati_mediana['saldo_banca_nominale'][idx_inizio_prelievo]
    patrimonio_etf_inizio = dati_mediana['saldo_etf_nominale'][idx_inizio_prelievo]
    patrimonio_fp_inizio = dati_mediana['saldo_fp_nominale'][idx_inizio_prelievo]
    
    print("=== ANALISI PATRIMONIO ALL'INIZIO PRELIEVI ===")
    print(f"Patrimonio banca: €{patrimonio_banca_inizio:,.0f}")
    print(f"Patrimonio ETF: €{patrimonio_etf_inizio:,.0f}")
    print(f"Patrimonio FP: €{patrimonio_fp_inizio:,.0f}")
    print(f"TOTALE: €{patrimonio_banca_inizio + patrimonio_etf_inizio + patrimonio_fp_inizio:,.0f}")
    print()
    
    # Calcolo teorico dei contributi
    contributi_banca_teorici = parametri_test['contributo_mensile_banca'] * 12 * parametri_test['anni_inizio_prelievo']
    contributi_etf_teorici = parametri_test['contributo_mensile_etf'] * 12 * parametri_test['anni_inizio_prelievo']
    contributi_fp_teorici = parametri_test['contributo_annuo_fp'] * parametri_test['anni_inizio_prelievo']
    
    print("=== CONTRIBUTI TEORICI (senza inflazione) ===")
    print(f"Contributi banca: €{contributi_banca_teorici:,.0f}")
    print(f"Contributi ETF: €{contributi_etf_teorici:,.0f}")
    print(f"Contributi FP: €{contributi_fp_teorici:,.0f}")
    print(f"TOTALE CONTRIBUTI: €{contributi_banca_teorici + contributi_etf_teorici + contributi_fp_teorici:,.0f}")
    print()
    
    # Analisi del patrimonio finale
    patrimonio_banca_finale = dati_mediana['saldo_banca_nominale'][-1]
    patrimonio_etf_finale = dati_mediana['saldo_etf_nominale'][-1]
    patrimonio_fp_finale = dati_mediana['saldo_fp_nominale'][-1]
    
    print("=== ANALISI PATRIMONIO FINALE ===")
    print(f"Patrimonio banca finale: €{patrimonio_banca_finale:,.0f}")
    print(f"Patrimonio ETF finale: €{patrimonio_etf_finale:,.0f}")
    print(f"Patrimonio FP finale: €{patrimonio_fp_finale:,.0f}")
    print(f"TOTALE FINALE: €{patrimonio_banca_finale + patrimonio_etf_finale + patrimonio_fp_finale:,.0f}")
    print()
    
    # Analisi dei prelievi
    prelievi_totali = np.sum(dati_mediana['prelievi_effettivi_nominali'])
    print("=== ANALISI PRELIEVI ===")
    print(f"Prelievi totali effettuati: €{prelievi_totali:,.0f}")
    print(f"Prelievo medio annuo: €{prelievi_totali / (parametri_test['anni_totali'] - parametri_test['anni_inizio_prelievo']):,.0f}")
    print()
    
    # Verifica della coerenza
    patrimonio_iniziale = parametri_test['capitale_iniziale'] + parametri_test['etf_iniziale']
    contributi_totali = stats['contributi_totali_versati_mediano_nominale']
    guadagni = stats['guadagni_accumulo_mediano_nominale']
    patrimonio_finale = stats['patrimonio_finale_mediano_nominale']
    
    print("=== VERIFICA COERENZA ===")
    print(f"Patrimonio iniziale: €{patrimonio_iniziale:,.0f}")
    print(f"+ Contributi totali: €{contributi_totali:,.0f}")
    print(f"+ Guadagni da investimento: €{guadagni:,.0f}")
    print(f"= Patrimonio teorico: €{patrimonio_iniziale + contributi_totali + guadagni:,.0f}")
    print(f"= Patrimonio finale effettivo: €{patrimonio_finale:,.0f}")
    print(f"Differenza: €{patrimonio_finale - (patrimonio_iniziale + contributi_totali + guadagni):,.0f}")
    print()
    
    # Analisi dei rendimenti annuali
    rendimenti_annuali = dati_mediana['rendimento_investimento_percentuale'][1:parametri_test['anni_inizio_prelievo']+1]
    print("=== RENDIMENTI ANNUALI (fase accumulo) ===")
    print(f"Rendimento medio: {np.mean(rendimenti_annuali):.2%}")
    print(f"Rendimento mediano: {np.median(rendimenti_annuali):.2%}")
    print(f"Rendimento min: {np.min(rendimenti_annuali):.2%}")
    print(f"Rendimento max: {np.max(rendimenti_annuali):.2%}")
    print(f"Anni negativi: {np.sum(rendimenti_annuali < 0)}/{len(rendimenti_annuali)}")
    print()
    
    # --- DEBUG ANNO PER ANNO ---
    print("\n=== DETTAGLIO ANNO PER ANNO (fase accumulo e prelievo) ===")
    for anno in range(1, parametri_test['anni_totali'] + 1):
        banca = dati_mediana['saldo_banca_nominale'][anno]
        etf = dati_mediana['saldo_etf_nominale'][anno]
        fp = dati_mediana['saldo_fp_nominale'][anno]
        prelievi = dati_mediana['prelievi_effettivi_nominali'][anno] if 'prelievi_effettivi_nominali' in dati_mediana else 0
        contributi = dati_mediana['contributi_totali_versati'][anno] - dati_mediana['contributi_totali_versati'][anno-1] if 'contributi_totali_versati' in dati_mediana else 0
        rendimento = dati_mediana['rendimento_investimento_percentuale'][anno] if 'rendimento_investimento_percentuale' in dati_mediana else 0
        print(f"Anno {anno:2d}: Banca €{banca:,.0f} | ETF €{etf:,.0f} | FP €{fp:,.0f} | Contributi €{contributi:,.0f} | Prelievi €{prelievi:,.0f} | Rend. {rendimento:+.2%}")
    print()
    
    return risultati

def test_economic_models():
    """Testa i modelli economici per verificare la realisticità dei rendimenti"""
    
    print("=== ANALISI MODELLI ECONOMICI ===\n")
    
    for model_name in engine.ECONOMIC_MODELS.keys():
        print(f"📊 MODELLO: {model_name}")
        print(f"Descrizione: {engine.ECONOMIC_MODELS[model_name]['description']}")
        
        model_params = engine._get_regime_params(model_name)
        market_regimes = model_params['market_regimes']
        
        print("\nRegimi di Mercato:")
        for regime_name, regime_params in market_regimes.items():
            mean_annual = regime_params['mean']
            vol_annual = regime_params['vol']
            transitions = regime_params.get('transitions', {})
            
            print(f"  • {regime_name}:")
            print(f"    - Rendimento medio annuo: {mean_annual:+.1%}")
            print(f"    - Volatilità annua: {vol_annual:.1%}")
            print(f"    - Transizioni: {transitions}")
            
            # Calcola rendimento medio mensile
            mean_monthly = mean_annual / 12
            vol_monthly = vol_annual / np.sqrt(12)
            
            # Simula 1000 mesi per vedere la distribuzione
            np.random.seed(42)  # Per riproducibilità
            rendimenti_mensili = np.random.normal(mean_monthly, vol_monthly, 1000)
            rendimenti_annuali = (1 + rendimenti_mensili) ** 12 - 1
            
            print(f"    - Rendimento medio simulato (1000 mesi): {np.mean(rendimenti_annuali):+.1%}")
            print(f"    - Percentile 5%: {np.percentile(rendimenti_annuali, 5):+.1%}")
            print(f"    - Percentile 95%: {np.percentile(rendimenti_annuali, 95):+.1%}")
            print(f"    - Anni negativi: {(rendimenti_annuali < 0).sum()}/1000 ({100*(rendimenti_annuali < 0).sum()/1000:.1f}%)")
            print()
        
        print("-" * 80)
        print()

def test_historical_comparison():
    """Confronta con dati storici realistici"""
    print("=== CONFRONTO CON DATI STORICI ===\n")
    
    # Dati storici S&P 500 (1928-2023)
    # Fonte: https://www.macrotrends.net/2526/sp-500-historical-annual-returns
    print("📈 Dati Storici S&P 500 (1928-2023):")
    print("  - Rendimento medio annuo: +10.2%")
    print("  - Volatilità annua: ~15.5%")
    print("  - Anni negativi: ~27% (26 su 96 anni)")
    print("  - Peggior anno: -43.8% (1931)")
    print("  - Miglior anno: +52.6% (1954)")
    print()
    
    print("🔍 PROBLEMI IDENTIFICATI NEI MODELLI:")
    print("  1. Regime 'Crash' con -40% annuo è troppo estremo")
    print("  2. Regime 'Recession' con -5% annuo è troppo pessimistico")
    print("  3. Le transizioni tra regimi potrebbero essere troppo frequenti")
    print("  4. Manca un regime 'Bull Market' con rendimenti positivi sostenuti")
    print()

def suggest_improvements():
    """Suggerisce miglioramenti ai modelli economici"""
    print("=== SUGGERIMENTI DI MIGLIORAMENTO ===\n")
    
    print("🎯 MODELLO VOLATILE MIGLIORATO:")
    print("""
    "VOLATILE (CICLI BOOM-BUST)": {
        "description": "Modello realistico con cicli di mercato basato su dati storici",
        "market_regimes": {
            'Bull Market': {'mean': 0.15, 'vol': 0.12, 'transitions': {'Bull Market': 0.95, 'Correction': 0.05}},
            'Correction': {'mean': -0.15, 'vol': 0.25, 'transitions': {'Bear Market': 0.7, 'Bull Market': 0.3}},
            'Bear Market': {'mean': -0.25, 'vol': 0.30, 'transitions': {'Bear Market': 0.8, 'Recovery': 0.2}},
            'Recovery': {'mean': 0.20, 'vol': 0.20, 'transitions': {'Bull Market': 0.9, 'Correction': 0.1}}
        },
        "inflation_regimes": {
            'Normal': {'mean': 0.025, 'vol': 0.01, 'transitions': {'Normal': 0.98, 'High': 0.02}},
            'High': {'mean': 0.06, 'vol': 0.03, 'transitions': {'Normal': 0.9, 'High': 0.1}}
        }
    }
    """)
    
    print("📊 VANTAGGI DEL MODELLO MIGLIORATO:")
    print("  - Rendimenti più realistici (max -25% invece di -40%)")
    print("  - Include un regime 'Bull Market' con +15% annuo")
    print("  - Transizioni più graduali tra i regimi")
    print("  - Percentuale di anni negativi più vicina ai dati storici")
    print()

if __name__ == "__main__":
    debug_single_simulation()
    test_economic_models()
    test_historical_comparison()
    suggest_improvements() 