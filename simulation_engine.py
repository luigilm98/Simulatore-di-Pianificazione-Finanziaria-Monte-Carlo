# -*- coding: utf-8 -*-
"""
Modulo Core per la Simulazione Finanziaria Monte Carlo.

Questo modulo contiene tutta la logica di calcolo per il simulatore di pianificazione
finanziaria. È progettato per essere indipendente da Streamlit o da qualsiasi altra
interfaccia utente.

La funzione principale è `run_full_simulation`, che orchestra l'esecuzione di
migliaia di singole traiettorie finanziarie e ne aggrega i risultati.

Il cuore del motore è un **Modello Economico a Regimi Commutabili (Regime-Switching Model)**.
Questo approccio supera i limiti di un semplice "random walk" introducendo diversi
"stati" o "regimi" per il mercato azionario e per l'inflazione (es. 'Normal', 'Crash',
'Recession', 'Recovery'). Ogni regime ha le sue caratteristiche di rendimento medio e
volatilità, e il modello definisce le probabilità di transizione da un regime all'altro,
creando cicli economici più realistici e correlati nel tempo.
"""

import numpy as np
import json

# ============================================================================
# DEFINIZIONE DEI MODELLI ECONOMICI A REGIMI
# ==============================================================================
ECONOMIC_MODELS = {
    "VOLATILE (CICLI BOOM-BUST)": {
        "description": "Un modello più realistico con cicli di mercato pronunciati. Le crisi sono seguite da recessioni e poi da forti riprese.",
        "market_regimes": {
            'Normal': {'mean': 0.08, 'vol': 0.15, 'transitions': {'Normal': 0.99, 'Crash': 0.01}},
            'Crash': {'mean': -0.40, 'vol': 0.50, 'transitions': {'Recession': 1.0}},
            'Recession': {'mean': -0.05, 'vol': 0.30, 'transitions': {'Recession': 0.95, 'Recovery': 0.05}},
            'Recovery': {'mean': 0.25, 'vol': 0.40, 'transitions': {'Normal': 1.0}}
        },
        "inflation_regimes": {
            'Normal': {'mean': 0.025, 'vol': 0.01, 'transitions': {'Normal': 0.98, 'High': 0.02}},
            'High': {'mean': 0.06, 'vol': 0.03, 'transitions': {'Normal': 0.9, 'High': 0.1}}
        },
        "initial_market_regime": "Normal",
        "initial_inflation_regime": "Normal"
    },
    "STAGFLAZIONE ANNI '70": {
        "description": "Simula un periodo di alta inflazione persistente con una crescita di mercato debole o negativa, simile agli anni '70.",
        "market_regimes": {
            'Stagnation': {'mean': 0.01, 'vol': 0.20, 'transitions': {'Stagnation': 0.98, 'WeakRecovery': 0.02}},
            'WeakRecovery': {'mean': 0.05, 'vol': 0.25, 'transitions': {'Stagnation': 1.0}}
        },
        "inflation_regimes": {
            'High': {'mean': 0.08, 'vol': 0.04, 'transitions': {'High': 0.99, 'VeryHigh': 0.01}},
            'VeryHigh': {'mean': 0.12, 'vol': 0.05, 'transitions': {'High': 1.0}}
        },
        "initial_market_regime": "Stagnation",
        "initial_inflation_regime": "High"
    },
    "CRESCITA STABILE (POST-2009)": {
        "description": "Un'era di crescita costante, bassa inflazione e volatilità contenuta, con rare e brevi flessioni.",
        "market_regimes": {
            'Growth': {'mean': 0.10, 'vol': 0.12, 'transitions': {'Growth': 0.995, 'Dip': 0.005}},
            'Dip': {'mean': -0.10, 'vol': 0.25, 'transitions': {'Growth': 1.0}}
        },
        "inflation_regimes": {
            'Low': {'mean': 0.02, 'vol': 0.005, 'transitions': {'Low': 1.0}}
        },
        "initial_market_regime": "Growth",
        "initial_inflation_regime": "Low"
    },
    "CRISI PROLUNGATA (GIAPPONE)": {
        "description": "Modella un 'decennio perduto' con mercati stagnanti per un lungo periodo e un rischio persistente di deflazione.",
        "market_regimes": {
            'Stagnation': {'mean': -0.01, 'vol': 0.18, 'transitions': {'Stagnation': 1.0}}
        },
        "inflation_regimes": {
            'Deflation': {'mean': -0.01, 'vol': 0.01, 'transitions': {'Deflation': 0.99, 'Normal': 0.01}},
            'Normal': {'mean': 0.01, 'vol': 0.01, 'transitions': {'Deflation': 1.0}}
        },
        "initial_market_regime": "Stagnation",
        "initial_inflation_regime": "Deflation"
    }
}

# ==============================================================================
# FUNZIONI HELPER PER IL MODELLO ECONOMICO
# ==============================================================================

def _get_regime_params(model_name):
    """
    Recupera i parametri per un dato modello economico dal dizionario globale.

    Se il nome del modello non viene trovato, restituisce di default il modello
    "VOLATILE (CICLI BOOM-BUST)".

    Args:
        model_name (str): Il nome del modello economico da caricare.

    Returns:
        dict: Un dizionario contenente la configurazione del modello.
    """
    return ECONOMIC_MODELS.get(model_name, ECONOMIC_MODELS["VOLATILE (CICLI BOOM-BUST)"])

def _choose_next_regime(current_regime, regime_definitions):
    """
    Determina il regime del mese successivo utilizzando una catena di Markov.

    Basandosi sul regime attuale, questa funzione estrae le probabilità di
    transizione verso altri regimi (o di rimanere nello stesso) e ne sceglie
    uno in modo probabilistico.

    Args:
        current_regime (str): Il nome del regime attuale (es. 'Normal', 'Crash').
        regime_definitions (dict): La parte del dizionario del modello economico
            che contiene le definizioni dei regimi (es. `market_regimes`).

    Returns:
        str: Il nome del regime scelto per il mese successivo.
    """
    if not regime_definitions.get(current_regime, {}).get('transitions'):
        return current_regime # Se non ci sono transizioni, rimane nello stesso stato
    transitions = regime_definitions[current_regime]['transitions']
    regimes, probs = zip(*transitions.items())
    return np.random.choice(regimes, p=probs)

def _calcola_sharpe_ratio_medio(tutti_i_dati_annuali):
    """
    Calcola lo Sharpe Ratio medio basato sulle variazioni percentuali annuali
    del patrimonio di tutte le simulazioni.
    
    Lo Sharpe Ratio è definito come: (Rendimento Medio - Tasso Risk-Free) / Deviazione Standard
    
    Args:
        tutti_i_dati_annuali (list): Lista di dizionari contenenti i dati annuali di ogni simulazione.
        
    Returns:
        float: Lo Sharpe Ratio medio calcolato.
    """
    if not tutti_i_dati_annuali:
        return 0.0
    
    # Raccogli tutte le variazioni percentuali annuali da tutte le simulazioni
    tutte_le_variazioni = []
    for dati_simulazione in tutti_i_dati_annuali:
        variazioni = dati_simulazione.get('variazione_patrimonio_percentuale', [])
        # Filtra valori validi (escludi NaN e infiniti)
        variazioni_valide = [v for v in variazioni if np.isfinite(v)]
        tutte_le_variazioni.extend(variazioni_valide)
    
    if not tutte_le_variazioni:
        return 0.0
    
    # Converti in array numpy
    variazioni_array = np.array(tutte_le_variazioni)
    
    # Calcola rendimento medio e deviazione standard
    rendimento_medio = np.mean(variazioni_array)
    deviazione_standard = np.std(variazioni_array)
    
    # Tasso risk-free (assumiamo 0% per semplicità, ma potrebbe essere parametrizzato)
    tasso_risk_free = 0.0
    
    # Calcola Sharpe Ratio
    if deviazione_standard > 0:
        sharpe_ratio = (rendimento_medio - tasso_risk_free) / deviazione_standard
    else:
        sharpe_ratio = 0.0
    
    return sharpe_ratio

# ==============================================================================
# FUNZIONI CORE DELLA SIMULAZIONE
# ==============================================================================

def valida_parametri(parametri):
    """
    Controlla la validità e la coerenza dei parametri di input della simulazione.

    Solleva un'eccezione `ValueError` se un parametro non rientra nei limiti
    attesi, per prevenire errori di calcolo.

    Args:
        parametri (dict): Il dizionario dei parametri inviato dall'interfaccia utente.
    """
    # Questa funzione è volutamente verbosa per avere messaggi di errore chiari.
    if parametri['eta_iniziale'] < 0:
        raise ValueError("Età iniziale non può essere negativa")
    if parametri['capitale_iniziale'] < 0:
        raise ValueError("Capitale iniziale non può essere negativo")
    if parametri['etf_iniziale'] < 0:
        raise ValueError("ETF iniziale non può essere negativo")
    if parametri['contributo_mensile_banca'] < 0:
        raise ValueError("Contributo mensile banca non può essere negativo")
    if parametri['contributo_mensile_etf'] < 0:
        raise ValueError("Contributo mensile ETF non può essere negativo")
    if parametri['anni_inizio_prelievo'] < 0:
        raise ValueError("Anni al prelievo non può essere negativo")
    if parametri['prelievo_annuo'] < 0:
        raise ValueError("Prelievo annuo non può essere negativo")
    if parametri['n_simulazioni'] <= 0:
        raise ValueError("Numero simulazioni deve essere positivo")
    if parametri['anni_totali'] <= 0:
        raise ValueError("Anni totali deve essere positivo")
    if not (0 <= parametri['tassazione_capital_gain'] <= 1):
        raise ValueError("La tassazione sul capital gain deve essere tra 0 e 1")
    if not (0 <= parametri['ter_etf'] <= 1):
        raise ValueError("Il TER degli ETF deve essere tra 0 e 1")
    if parametri['costo_fisso_etf_mensile'] < 0:
        raise ValueError("Il costo fisso ETF mensile non può essere negativo")
    if parametri['attiva_fondo_pensione']:
        if not (0 <= parametri['rendimento_medio_fp'] <= 1):
            raise ValueError("Rendimento medio FP deve essere tra 0 e 1")
        if not (0 <= parametri['ter_fp'] <= 1):
            raise ValueError("TER FP deve essere tra 0 e 1")
        if not (0 <= parametri['aliquota_finale_fp'] <= 1):
            raise ValueError("Aliquota finale FP deve essere tra 0 e 1 (es. 0.15 per 15%)")
    
    # Validazione parametri ribilanciamento
    if parametri.get('strategia_ribilanciamento', 'GLIDEPATH') == 'GLIDEPATH':
        inizio_glidepath = parametri.get('inizio_glidepath_anni', 20)
        fine_glidepath = parametri.get('fine_glidepath_anni', 40)
        if inizio_glidepath >= fine_glidepath:
            raise ValueError("Inizio glidepath deve essere prima della fine")
        if fine_glidepath > parametri.get('anni_totali', 40):
            raise ValueError("Fine glidepath oltre l'orizzonte temporale")
    
    # Validazione parametri costi e tasse
    if not (0 <= parametri.get('imposta_bollo_titoli', 0.002) <= 1):
        raise ValueError("Imposta di bollo titoli deve essere tra 0 e 1")
    if parametri.get('imposta_bollo_conto', 34.20) < 0:
        raise ValueError("Imposta di bollo conto non può essere negativa")
    
    # Validazione strategia prelievo
    if parametri.get('strategia_prelievo', 'REGOLA_4_PERCENTO') == 'GUARDRAIL':
        if not (0 <= parametri.get('banda_guardrail', 0.10) <= 1):
            raise ValueError("Banda guardrail deve essere tra 0 e 1")

def _calcola_allocazione_annuale(parametri):
    """
    Calcola l'allocazione ETF/liquidità per ogni anno in base alla strategia di ribilanciamento.
    
    Args:
        parametri: Parametri della simulazione
        
    Returns:
        Array con allocazione ETF per ogni anno (0-1)
    """
    anni_totali = parametri.get('anni_totali', 40)
    strategia_ribilanciamento = parametri.get('strategia_ribilanciamento', 'GLIDEPATH')
    
    # Allocazione iniziale basata sui valori iniziali
    capitale_iniziale = parametri.get('capitale_iniziale', 0)
    etf_iniziale = parametri.get('etf_iniziale', 0)
    patrimonio_totale_iniziale = capitale_iniziale + etf_iniziale
    
    if patrimonio_totale_iniziale > 0:
        allocazione_iniziale = etf_iniziale / patrimonio_totale_iniziale
    else:
        allocazione_iniziale = 0.60  # Default 60% ETF, 40% liquidità
    
    allocazioni_annuali = np.zeros(anni_totali)
    
    if strategia_ribilanciamento == 'GLIDEPATH':
        # Glidepath: riduzione progressiva del rischio
        inizio_glidepath = parametri.get('inizio_glidepath_anni', 20)
        fine_glidepath = parametri.get('fine_glidepath_anni', 40)
        allocazione_finale = parametri.get('allocazione_etf_finale', 0.333)
        
        for anno in range(anni_totali):
            if anno < inizio_glidepath:
                # Fase accumulo: allocazione costante
                allocazioni_annuali[anno] = allocazione_iniziale
            elif anno >= fine_glidepath:
                # Fase finale: allocazione target
                allocazioni_annuali[anno] = allocazione_finale
            else:
                # Fase transizione: riduzione lineare
                progresso = (anno - inizio_glidepath) / (fine_glidepath - inizio_glidepath)
                allocazioni_annuali[anno] = allocazione_iniziale + progresso * (allocazione_finale - allocazione_iniziale)
                
    elif strategia_ribilanciamento == 'ANNUALE_FISSO':
        # Ribilanciamento annuale a allocazione fissa
        allocazione_fissa = parametri.get('allocazione_etf_fissa', 0.60)
        allocazioni_annuali[:] = allocazione_fissa
        
    else:  # NESSUNO
        # Nessun ribilanciamento: allocazione iniziale mantenuta
        allocazioni_annuali[:] = allocazione_iniziale
    
    return allocazioni_annuali

def _esegui_una_simulazione(parametri, prelievo_annuo_da_usare):
    """
    Esegue una singola traiettoria di simulazione finanziaria.
    Questa funzione è stata completamente riscritta per garantire la correttezza contabile.
    """
    # --- 1. SETUP INIZIALE ---
    np.random.seed()
    num_anni = parametri['anni_totali']
    mesi_totali = num_anni * 12
    inizio_prelievo_mesi = parametri['anni_inizio_prelievo'] * 12

    # Inizializzazione dei contenitori per i dati annuali
    dati_annuali = {k: np.zeros(num_anni + 1) for k in [
        'saldo_banca_nominale', 'saldo_etf_nominale', 'saldo_fp_nominale',
        'saldo_banca_reale', 'saldo_etf_reale', 'saldo_fp_reale',
        'stipendi_netti_nominali',
        'prelievi_target_nominali', 'prelievi_effettivi_nominali', 'prelievi_effettivi_reali',
        'prelievi_da_banca_nominali', 'prelievi_da_etf_nominali',
        'pensioni_pubbliche_nominali', 'pensioni_pubbliche_reali',
        'rendite_fp_nominali', 'rendite_fp_reali',
        'fp_liquidato_nominale', 'fp_liquidato_reale',
        'variazione_patrimonio_percentuale', 'rendimento_investimento_percentuale',
        'contributi_totali_versati', 'indice_prezzi', 'reddito_totale_reale',
        'vendite_rebalance_nominali'
    ]}

    # Stato iniziale dei saldi e delle variabili
    patrimonio_banca = parametri['capitale_iniziale']
    patrimonio_etf = parametri['etf_iniziale']
    etf_cost_basis = patrimonio_etf
    patrimonio_fp = 0
    contributi_totali_fp = 0
    etf_cashflow_anno = 0.0
    
    dati_annuali['saldo_banca_nominale'][0] = patrimonio_banca
    dati_annuali['saldo_etf_nominale'][0] = patrimonio_etf
    dati_annuali['indice_prezzi'][0] = 1.0

    # Variabili di stato della simulazione
    indice_prezzi = 1.0
    contributi_totali_accumulati = 0
    guadagni_accumulo = 0
    guadagni_calcolati = False
    
    prelievo_annuo_nominale_corrente = 0.0
    prelievo_annuo_nominale_iniziale = 0.0
    indice_prezzi_inizio_pensione = 1.0

    # Variabili di stato per la gestione della rendita FP
    rendita_fp_mese = 0
    rendita_fp_mese_iniziale = 0
    mesi_rimanenti_rendita_fp = 0

    # Modello economico a regimi
    model_name = parametri.get('economic_model', "VOLATILE (CICLI BOOM-BUST)")
    economic_model_params = _get_regime_params(model_name)
    market_regime_definitions = economic_model_params['market_regimes']
    inflation_regime_definitions = economic_model_params['inflation_regimes']
    current_market_regime = np.random.choice(list(market_regime_definitions.keys()))
    current_inflation_regime = np.random.choice(list(inflation_regime_definitions.keys()))

    # --- LOGICA COMBINAZIONE PARAMETRI RENDIMENTO ---
    modalita_parametri = parametri.get('modalita_parametri_rendimento', 'Combinazione Pesata')
    peso_azioni = parametri.get('peso_azioni', 0.6)  # Default 60% azioni se non specificato
    rendimento_portafoglio = parametri.get('rendimento_medio', 0.06)
    volatilita_portafoglio = parametri.get('volatilita', 0.12)

    # --- 2. LOOP DI SIMULAZIONE MENSILE ---
    for mese in range(1, mesi_totali + 1):
        anno_corrente = (mese - 1) // 12 + 1
        eta_attuale = parametri['eta_iniziale'] + (mese - 1) / 12

        # A. GESTIONE EVENTI E FONDO PENSIONE
        if parametri.get('attiva_fondo_pensione', False):
            # Evento di liquidazione all'età di ritiro (eseguito solo una volta)
            if int(eta_attuale) == parametri.get('eta_ritiro_fp', 67) and mese % 12 == 1 and patrimonio_fp > 0:
                guadagni_fp = patrimonio_fp - contributi_totali_fp
                tasse_fp = max(0, guadagni_fp) * parametri.get('aliquota_finale_fp', 0.15)
                patrimonio_fp_netto = patrimonio_fp - tasse_fp
                
                percentuale_capitale = parametri.get('percentuale_capitale_fp', 0.5)
                capitale_liquidato = patrimonio_fp_netto * percentuale_capitale
                importo_per_rendita = patrimonio_fp_netto - capitale_liquidato
                
                patrimonio_banca += capitale_liquidato
                
                # Salva la liquidazione FP nell'anno corrente (sia nominale che reale)
                dati_annuali['fp_liquidato_nominale'][anno_corrente] += capitale_liquidato
                dati_annuali['fp_liquidato_reale'][anno_corrente] += capitale_liquidato / indice_prezzi
                
                durata_rendita_anni = parametri.get('durata_rendita_fp_anni', 25)
                if durata_rendita_anni > 0:
                    mesi_rimanenti_rendita_fp = durata_rendita_anni * 12
                    # Calcola rendita mensile iniziale (verrà rivalutata per inflazione)
                    rendita_fp_mese_iniziale = importo_per_rendita / mesi_rimanenti_rendita_fp if mesi_rimanenti_rendita_fp > 0 else 0
                    rendita_fp_mese = rendita_fp_mese_iniziale
                
                patrimonio_fp = 0 # Il fondo viene azzerato

            # Erogazione della rendita mensile (rivalutata per inflazione)
            if mesi_rimanenti_rendita_fp > 0:
                # Rivaluta la rendita per inflazione
                rendita_fp_mese = rendita_fp_mese_iniziale * indice_prezzi
                mesi_rimanenti_rendita_fp -= 1
            
            if mesi_rimanenti_rendita_fp == 0:
                rendita_fp_mese = 0
        
        # B. ENTRATE MENSILI E AGGIORNAMENTO DATI
        # Calcolo Pensione Pubblica
        pensione_pubblica_mese = 0
        inizio_pensione_mesi = parametri.get('inizio_pensione_anni', num_anni + 1) * 12
        if mese >= inizio_pensione_mesi:
            # La pensione pubblica impostata dall'utente è in termini reali
            # Deve essere rivalutata per inflazione per mantenere il potere d'acquisto
            pensione_annua_reale = parametri.get('pensione_pubblica_annua', 0)
            pensione_annua_nominale = pensione_annua_reale * indice_prezzi
            pensione_pubblica_mese = pensione_annua_nominale / 12
        
        # Aggiornamento contabile: accredito entrate e salvataggio dati
        patrimonio_banca += pensione_pubblica_mese + rendita_fp_mese
        
        dati_annuali['pensioni_pubbliche_nominali'][anno_corrente] += pensione_pubblica_mese
        dati_annuali['pensioni_pubbliche_reali'][anno_corrente] += pensione_pubblica_mese / indice_prezzi
        dati_annuali['rendite_fp_nominali'][anno_corrente] += rendita_fp_mese
        dati_annuali['rendite_fp_reali'][anno_corrente] += rendita_fp_mese / indice_prezzi
        
        reddito_da_pensioni_reale = (pensione_pubblica_mese + rendita_fp_mese) / indice_prezzi
        dati_annuali['reddito_totale_reale'][anno_corrente] += reddito_da_pensioni_reale

        # C. FASE DI ACCUMULO (prima dei rendimenti)
        if mese < inizio_prelievo_mesi:
            # Rivaluta i contributi per l'inflazione corrente
            contributo_mensile_banca_nominale = parametri['contributo_mensile_banca'] * indice_prezzi
            contributo_mensile_etf_nominale = parametri['contributo_mensile_etf'] * indice_prezzi

            patrimonio_banca += contributo_mensile_banca_nominale
            contributi_totali_accumulati += contributo_mensile_banca_nominale
            
            investimento_etf = min(contributo_mensile_etf_nominale, patrimonio_banca)
            if investimento_etf > 0:
                patrimonio_banca -= investimento_etf
                patrimonio_etf += investimento_etf
                etf_cost_basis += investimento_etf
                contributi_totali_accumulati += investimento_etf
                etf_cashflow_anno += investimento_etf

        # D. FASE DI PRELIEVO (prima dei rendimenti)
        if mese >= inizio_prelievo_mesi:
            # Calcolo fabbisogno reale e nominale
            if not guadagni_calcolati:
                patrimonio_attuale = patrimonio_banca + patrimonio_etf + patrimonio_fp
                guadagni_accumulo = patrimonio_attuale - (parametri['capitale_iniziale'] + parametri['etf_iniziale']) - contributi_totali_accumulati
                guadagni_calcolati = True

            # Imposta/aggiorna il prelievo annuale SOLO UNA VOLTA ALL'ANNO
            if (mese - inizio_prelievo_mesi) % 12 == 0:
                if parametri['strategia_prelievo'] == 'FISSO':
                    prelievo_annuo_nominale_corrente = prelievo_annuo_da_usare * indice_prezzi
                elif parametri['strategia_prelievo'] == 'REGOLA_4_PERCENTO':
                    if mese == inizio_prelievo_mesi:
                        patrimonio_a_prelievo = patrimonio_banca + patrimonio_etf
                        prelievo_annuo_nominale_iniziale = patrimonio_a_prelievo * parametri['percentuale_regola_4']
                        indice_prezzi_inizio_pensione = indice_prezzi
                        prelievo_annuo_nominale_corrente = prelievo_annuo_nominale_iniziale
                    else:
                        prelievo_annuo_nominale_corrente = prelievo_annuo_nominale_iniziale * (indice_prezzi / indice_prezzi_inizio_pensione)
                elif parametri['strategia_prelievo'] == 'GUARDRAIL':
                    if mese == inizio_prelievo_mesi:
                        patrimonio_a_prelievo = patrimonio_banca + patrimonio_etf
                        prelievo_annuo_nominale_iniziale = patrimonio_a_prelievo * parametri['percentuale_regola_4']
                        indice_prezzi_inizio_pensione = indice_prezzi
                        prelievo_annuo_nominale_corrente = prelievo_annuo_nominale_iniziale
                    else:
                        # Calcola trend di mercato (ultimi 3 anni)
                        anni_da_prelievo = (mese - inizio_prelievo_mesi) // 12
                        if anni_da_prelievo >= 3:
                            # Calcola trend basato sul patrimonio attuale vs iniziale
                            patrimonio_attuale = patrimonio_banca + patrimonio_etf
                            trend_mercato = patrimonio_attuale / (patrimonio_a_prelievo * (indice_prezzi / indice_prezzi_inizio_pensione))
                            
                            banda_guardrail = parametri.get('banda_guardrail', 0.10)
                            if trend_mercato > (1 + banda_guardrail):
                                # Mercato in forte crescita: aumenta prelievo
                                prelievo_annuo_nominale_corrente = prelievo_annuo_nominale_iniziale * (indice_prezzi / indice_prezzi_inizio_pensione) * (1 + banda_guardrail * 0.5)
                            elif trend_mercato < (1 - banda_guardrail):
                                # Mercato in calo: riduce prelievo
                                prelievo_annuo_nominale_corrente = prelievo_annuo_nominale_iniziale * (indice_prezzi / indice_prezzi_inizio_pensione) * (1 - banda_guardrail * 0.5)
                            else:
                                # Mercato stabile: prelievo normale
                                prelievo_annuo_nominale_corrente = prelievo_annuo_nominale_iniziale * (indice_prezzi / indice_prezzi_inizio_pensione)
                        else:
                            # Primi anni: prelievo normale
                            prelievo_annuo_nominale_corrente = prelievo_annuo_nominale_iniziale * (indice_prezzi / indice_prezzi_inizio_pensione)
            
            prelievo_mensile_target = prelievo_annuo_nominale_corrente / 12 if prelievo_annuo_nominale_corrente > 0 else 0
            if prelievo_mensile_target > 0:
                prelevato_da_banca = min(prelievo_mensile_target, patrimonio_banca)
                patrimonio_banca -= prelevato_da_banca
                
                fabbisogno_da_etf = prelievo_mensile_target - prelevato_da_banca
                prelevato_da_etf_netto = 0
                if fabbisogno_da_etf > 0 and patrimonio_etf > 0:
                    cost_basis_ratio = etf_cost_basis / patrimonio_etf if patrimonio_etf > 0 else 1.0
                    tasse_implicite = (1 - cost_basis_ratio) * parametri['tassazione_capital_gain']
                    importo_lordo_da_vendere = fabbisogno_da_etf / (1 - tasse_implicite) if (1 - tasse_implicite) > 0 else float('inf')
                    importo_venduto = min(importo_lordo_da_vendere, patrimonio_etf)
                    etf_cashflow_anno -= importo_venduto # Traccia il flusso in uscita
                    
                    if importo_venduto > 0:
                        costo_proporzionale = (importo_venduto / patrimonio_etf) * etf_cost_basis
                        plusvalenza = importo_venduto - costo_proporzionale
                        tasse = plusvalenza * parametri['tassazione_capital_gain']
                        prelevato_da_etf_netto = importo_venduto - tasse
                        
                        patrimonio_etf -= importo_venduto
                        etf_cost_basis -= costo_proporzionale

                prelievo_totale_mese = prelevato_da_banca + prelevato_da_etf_netto
                dati_annuali['prelievi_target_nominali'][anno_corrente] += prelievo_mensile_target
                dati_annuali['prelievi_effettivi_nominali'][anno_corrente] += prelievo_totale_mese
                dati_annuali['prelievi_effettivi_reali'][anno_corrente] += prelievo_totale_mese / indice_prezzi
                dati_annuali['prelievi_da_banca_nominali'][anno_corrente] += prelevato_da_banca
                dati_annuali['prelievi_da_etf_nominali'][anno_corrente] += prelevato_da_etf_netto
                
                dati_annuali['reddito_totale_reale'][anno_corrente] += prelievo_totale_mese / indice_prezzi

        # E. RENDIMENTI, COSTI E AGGIORNAMENTO INFLAZIONE
        market_regime = market_regime_definitions[current_market_regime]
        inflation_regime = inflation_regime_definitions[current_inflation_regime]

        # --- SCEGLI I PARAMETRI DI RENDIMENTO/VOlATILITÀ DA USARE ---
        if modalita_parametri == 'Solo Modello Economico':
            mean_mese = market_regime['mean'] / 12
            vol_mese = market_regime['vol'] / np.sqrt(12)
        elif modalita_parametri == 'Solo Portafoglio ETF':
            mean_mese = rendimento_portafoglio / 12
            vol_mese = volatilita_portafoglio / np.sqrt(12)
        else:  # Combinazione Pesata
            mean_mese = (peso_azioni * market_regime['mean'] + (1 - peso_azioni) * rendimento_portafoglio) / 12
            vol_mese = (peso_azioni * market_regime['vol'] + (1 - peso_azioni) * volatilita_portafoglio) / np.sqrt(12)

        rendimento_mensile = np.random.normal(mean_mese, vol_mese)
        inflazione_mensile = np.random.normal(inflation_regime['mean'] / 12, inflation_regime['vol'] / np.sqrt(12))
        
        patrimonio_etf *= (1 + rendimento_mensile)
        patrimonio_etf -= patrimonio_etf * (parametri['ter_etf'] / 12)
        
        # Applica costo fisso ETF mensile
        costo_fisso_mensile = parametri.get('costo_fisso_etf_mensile', 0.0)
        if costo_fisso_mensile > 0:
            patrimonio_banca -= costo_fisso_mensile
        
        # Applica imposte di bollo (annuali, a fine anno)
        if mese % 12 == 0:
            # Imposta di bollo titoli
            if patrimonio_etf > 0:
                imposta_bollo_titoli = patrimonio_etf * parametri.get('imposta_bollo_titoli', 0.002)
                patrimonio_etf -= imposta_bollo_titoli
            
            # Imposta di bollo conto (se giacenza > 5000€)
            if patrimonio_banca > 5000:
                imposta_bollo_conto = parametri.get('imposta_bollo_conto', 34.20)
                patrimonio_banca -= imposta_bollo_conto
        
        indice_prezzi *= (1 + inflazione_mensile)

        current_market_regime = _choose_next_regime(current_market_regime, market_regime_definitions)
        current_inflation_regime = _choose_next_regime(current_inflation_regime, inflation_regime_definitions)
        
        # F. RIBILANCIAMENTO ANNUALE (eccetto strategia NESSUNO)
        if mese % 12 == 0 and parametri.get('strategia_ribilanciamento', 'GLIDEPATH') != 'NESSUNO':
            allocazioni_annuali = _calcola_allocazione_annuale(parametri)
            allocazione_target = allocazioni_annuali[anno_corrente - 1] if anno_corrente > 0 else allocazioni_annuali[0]
            
            patrimonio_totale = patrimonio_banca + patrimonio_etf
            patrimonio_target_etf = patrimonio_totale * allocazione_target
            
            # Calcolo trasferimenti per ribilanciamento
            if patrimonio_etf > patrimonio_target_etf:
                # Troppo ETF: vendo ETF per comprare liquidità
                trasferimento = patrimonio_etf - patrimonio_target_etf
                
                # Calcola tasse sul capital gain
                if patrimonio_etf > 0 and etf_cost_basis > 0:
                    costo_proporzionale = (trasferimento / patrimonio_etf) * etf_cost_basis
                    plusvalenza = trasferimento - costo_proporzionale
                    tasse_rebalance = max(0, plusvalenza) * parametri['tassazione_capital_gain']
                    
                    patrimonio_etf -= trasferimento
                    patrimonio_banca += trasferimento - tasse_rebalance
                    etf_cost_basis -= costo_proporzionale
                    
                    # Track vendite di ribilanciamento
                    dati_annuali['vendite_rebalance_nominali'][anno_corrente] += trasferimento
                else:
                    patrimonio_etf -= trasferimento
                    patrimonio_banca += trasferimento
                    dati_annuali['vendite_rebalance_nominali'][anno_corrente] += trasferimento
            
            elif patrimonio_etf < patrimonio_target_etf:
                # Troppo liquidità: vendo liquidità per comprare ETF
                trasferimento = patrimonio_target_etf - patrimonio_etf
                patrimonio_banca -= trasferimento
                patrimonio_etf += trasferimento
                etf_cost_basis += trasferimento  # Aggiorna cost basis
        
        # G. OPERAZIONI DI FINE ANNO
        if mese % 12 == 0:
            # Crescita annuale e contributo al fondo pensione (se attivo)
            if parametri.get('attiva_fondo_pensione', False):
                # La crescita viene applicata solo se il fondo non è stato ancora liquidato
                if patrimonio_fp > 0:
                    rendimento_fp = np.random.normal(
                        parametri.get('rendimento_medio_fp', 0.04),
                        parametri.get('volatilita_fp', 0.08)
                    )
                    patrimonio_fp *= (1 + rendimento_fp)
                    patrimonio_fp -= patrimonio_fp * parametri.get('ter_fp', 0.01)
                    
                    # Applica tassazione sui rendimenti (se configurata)
                    tassazione_rendimenti_fp = parametri.get('tassazione_rendimenti_fp', 0.20)
                    if tassazione_rendimenti_fp > 0:
                        rendimento_netto = patrimonio_fp - contributi_totali_fp
                        if rendimento_netto > 0:
                            tasse_rendimenti = rendimento_netto * tassazione_rendimenti_fp
                            patrimonio_fp -= tasse_rendimenti
                
                # Il contributo viene aggiunto durante tutta la fase di accumulo
                if anno_corrente < parametri['anni_inizio_prelievo']:
                    contributo_fp = parametri.get('contributo_annuo_fp', 0)
                    patrimonio_fp += contributo_fp
                    contributi_totali_fp += contributo_fp

            patrimonio_inizio_anno = dati_annuali['saldo_banca_nominale'][anno_corrente-1] + dati_annuali['saldo_etf_nominale'][anno_corrente-1]
            patrimonio_fine_anno = patrimonio_banca + patrimonio_etf
            
            dati_annuali['variazione_patrimonio_percentuale'][anno_corrente] = (patrimonio_fine_anno - patrimonio_inizio_anno) / patrimonio_inizio_anno if patrimonio_inizio_anno > 0 else 0
            dati_annuali['saldo_banca_nominale'][anno_corrente] = patrimonio_banca
            dati_annuali['saldo_etf_nominale'][anno_corrente] = patrimonio_etf
            dati_annuali['saldo_fp_nominale'][anno_corrente] = patrimonio_fp
            dati_annuali['saldo_banca_reale'][anno_corrente] = patrimonio_banca / indice_prezzi
            dati_annuali['saldo_etf_reale'][anno_corrente] = patrimonio_etf / indice_prezzi
            dati_annuali['saldo_fp_reale'][anno_corrente] = patrimonio_fp / indice_prezzi
            dati_annuali['indice_prezzi'][anno_corrente] = indice_prezzi
            dati_annuali['contributi_totali_versati'][anno_corrente] = contributi_totali_accumulati
            
            # Calcolo rendimento puro degli investimenti con metodo Simple Dietz approssimato
            patrimonio_investimenti_inizio = dati_annuali['saldo_etf_nominale'][anno_corrente-1]
            patrimonio_investimenti_fine = patrimonio_etf
            
            # Denominatore: capitale iniziale + metà dei flussi di cassa netti dell'anno
            denominatore = patrimonio_investimenti_inizio + (etf_cashflow_anno / 2)
            
            if denominatore != 0:
                # Numeratore: guadagno/perdita netta (variazione di valore - flussi di cassa)
                guadagno_netto = patrimonio_investimenti_fine - patrimonio_investimenti_inizio - etf_cashflow_anno
                dati_annuali['rendimento_investimento_percentuale'][anno_corrente] = guadagno_netto / denominatore
            else:
                dati_annuali['rendimento_investimento_percentuale'][anno_corrente] = 0.0
            
            # Resetta il contatore dei flussi per l'anno successivo
            etf_cashflow_anno = 0.0

    # --- 3. OUTPUT FINALE ---
    patrimonio_storico = dati_annuali['saldo_banca_nominale'] + dati_annuali['saldo_etf_nominale']
    drawdown = 0
    if np.any(patrimonio_storico > 0):
        picchi = np.maximum.accumulate(patrimonio_storico)
        drawdown_values = (patrimonio_storico - picchi) / picchi
        drawdown = np.min(drawdown_values)

    return {
        "dati_annuali": dati_annuali,
        "drawdown": drawdown,
        "fallimento": (patrimonio_banca + patrimonio_etf) <= 0 and mese >= inizio_prelievo_mesi,
        "guadagni_accumulo": guadagni_accumulo,
        "contributi_totali_versati": contributi_totali_accumulati
    }


def _calcola_prelievo_sostenibile(parametri):
    """
    Trova il prelievo annuo reale massimo sostenibile con una ricerca binaria.
    L'obiettivo è trovare il tasso di prelievo che porta il patrimonio reale
    finale *mediano* il più vicino possibile a zero, senza diventare negativo.
    Utilizza un numero ridotto di simulazioni per la velocità.
    """
    params_test = parametri.copy()
    params_test['prelievo_annuo'] = 0
    params_test['n_simulazioni'] = max(100, params_test['n_simulazioni'] // 4)
    params_test['_in_routine_sostenibile'] = True  # Flag per evitare ricorsione

    risultati_test = run_full_simulation(params_test)
    patrimoni_reali_test = risultati_test['dati_grafici_principali']['reale']
    
    idx_inizio_prelievo = parametri['anni_inizio_prelievo']
    capitale_reale_mediano_a_prelievo = np.median(patrimoni_reali_test[:, idx_inizio_prelievo])

    anni_prelievo = parametri['anni_totali'] - parametri['anni_inizio_prelievo']
    if anni_prelievo <= 0:
        return 0
    # Se il capitale mediano è basso ma non nullo, prova comunque a cercare un prelievo >0
    if capitale_reale_mediano_a_prelievo <= 0:
        return 0
    
    # Range più ampio per la ricerca binaria
    limite_inferiore = 0
    limite_superiore = max(1, capitale_reale_mediano_a_prelievo * 2 / max(1, anni_prelievo))
    prelievo_ottimale = 0
    soglia = 10000  # Patrimonio finale massimo accettato
    max_fallimento = 0.10  # Probabilità di fallimento massima accettata
    
    def mediana_finale(prelievo):
        params_run = parametri.copy()
        params_run['prelievo_annuo'] = prelievo  # Assicura che venga usato il valore corretto
        params_run['n_simulazioni'] = max(100, params_run['n_simulazioni'] // 4)
        params_run['_in_routine_sostenibile'] = True  # Flag per evitare ricorsione
        risultati_run = run_full_simulation(params_run, prelievo_annuo_da_usare=prelievo)
        patrimonio_finale = np.median(risultati_run['dati_grafici_principali']['reale'][:, -1])
        prob_fallimento = risultati_run['statistiche']['probabilita_fallimento']
        return patrimonio_finale, prob_fallimento

    for _ in range(50):
        prelievo_corrente = (limite_inferiore + limite_superiore) / 2
        if prelievo_corrente < 1:
            break
        patrimonio_risultante, prob_fallimento = mediana_finale(prelievo_corrente)
        if abs(patrimonio_risultante) <= soglia and prob_fallimento <= max_fallimento:
            prelievo_ottimale = prelievo_corrente
            limite_inferiore = prelievo_corrente
        else:
            limite_superiore = prelievo_corrente
                
    return prelievo_ottimale


def run_full_simulation(parametri, prelievo_annuo_da_usare=None):
    valida_parametri(parametri)
    
    # Gestione del calcolo del prelievo sostenibile
    prelievo_sostenibile_calcolato = None
    if prelievo_annuo_da_usare is None:
        if parametri['strategia_prelievo'] == 'FISSO' and parametri.get('calcola_prelievo_sostenibile', False) and not parametri.get('_in_routine_sostenibile', False):
            prelievo_sostenibile_calcolato = _calcola_prelievo_sostenibile(parametri)
            prelievo_annuo_da_usare = prelievo_sostenibile_calcolato
            parametri['prelievo_annuo'] = prelievo_sostenibile_calcolato
            parametri['_in_routine_sostenibile'] = True  # Evita ricorsione
            # NON fare return! Prosegui con la simulazione principale usando il valore trovato
        else:
            prelievo_annuo_da_usare = parametri['prelievo_annuo']

    # Inizializzazione contenitori per i risultati aggregati
    n_sim = parametri['n_simulazioni']
    num_anni = parametri['anni_totali']
    
    # FIX: Inizializza un array per contenere tutti i dati annuali di tutte le run
    tutti_i_dati_annuali = [{} for _ in range(n_sim)]
    tutti_i_drawdown = np.zeros(n_sim)
    tutti_i_guadagni = np.zeros(n_sim)
    tutti_i_contributi = np.zeros(n_sim)
    fallimenti = 0

    # Esecuzione delle N simulazioni
    for i in range(n_sim):
        risultati_run = _esegui_una_simulazione(parametri, prelievo_annuo_da_usare)
        tutti_i_dati_annuali[i] = risultati_run['dati_annuali']
        tutti_i_drawdown[i] = risultati_run['drawdown']
        tutti_i_guadagni[i] = risultati_run['guadagni_accumulo']
        tutti_i_contributi[i] = risultati_run['contributi_totali_versati']
        if risultati_run['fallimento']:
            fallimenti += 1
            
    # --- Calcolo dei valori reali e nominali per i grafici ---
    patrimoni_nominali_tutte_le_run = np.array([
        d['saldo_banca_nominale'] + d['saldo_etf_nominale'] + d['saldo_fp_nominale'] 
        for d in tutti_i_dati_annuali
    ])
    
    patrimoni_reali_tutte_le_run = np.zeros_like(patrimoni_nominali_tutte_le_run)
    for i in range(n_sim):
        indici_prezzi = tutti_i_dati_annuali[i]['indice_prezzi']
        indici_prezzi = np.maximum(indici_prezzi, 1e-10) # Safety check
        patrimoni_reali_tutte_le_run[i, :] = patrimoni_nominali_tutte_le_run[i, :] / indici_prezzi

    # Identificazione dello scenario mediano basato sul patrimonio finale reale
    patrimoni_finali_reali = patrimoni_reali_tutte_le_run[:, -1]
    patrimoni_finali_reali = np.nan_to_num(patrimoni_finali_reali, nan=0.0, posinf=0.0, neginf=0.0)
    
    valore_mediano = np.median(patrimoni_finali_reali)
    indice_mediano = np.abs(patrimoni_finali_reali - valore_mediano).argmin() if len(patrimoni_finali_reali) > 0 else 0
    dati_mediana_dettagliati = tutti_i_dati_annuali[indice_mediano]

    # Calcolo delle statistiche aggregate finali
    patrimoni_finali_nominali = patrimoni_nominali_tutte_le_run[:, -1]
    idx_inizio_prelievo = parametri['anni_inizio_prelievo']
    
    statistiche = {
        'patrimonio_finale_mediano_nominale': np.median(patrimoni_finali_nominali),
        'patrimonio_finale_top_10_nominale': np.percentile(patrimoni_finali_nominali, 90),
        'patrimonio_finale_peggior_10_nominale': np.percentile(patrimoni_finali_nominali, 10),
        'patrimonio_finale_mediano_reale': valore_mediano,
        'patrimonio_finale_top_10_reale': np.percentile(patrimoni_finali_reali, 90),
        'patrimonio_finale_peggior_10_reale': np.percentile(patrimoni_finali_reali, 10),
        'patrimonio_inizio_prelievi_mediano_nominale': np.median(patrimoni_nominali_tutte_le_run[:, idx_inizio_prelievo]),
        'patrimonio_inizio_prelievi_mediano_reale': np.median(patrimoni_reali_tutte_le_run[:, idx_inizio_prelievo]),
        'probabilita_fallimento': fallimenti / n_sim if n_sim > 0 else 0,
        'drawdown_massimo_peggiore': np.min(tutti_i_drawdown) if len(tutti_i_drawdown) > 0 else 0,
        'sharpe_ratio_medio': _calcola_sharpe_ratio_medio(tutti_i_dati_annuali),
        'patrimoni_reali_finali': patrimoni_finali_reali,
        'guadagni_accumulo_mediano_nominale': np.median(tutti_i_guadagni),
        'contributi_totali_versati_mediano_nominale': np.median(tutti_i_contributi),
        'prelievo_sostenibile_calcolato': prelievo_sostenibile_calcolato
    }

    # Estrazione redditi per analisi
    reddito_reale_annuo_tutte_le_run = np.array([run['reddito_totale_reale'] for run in tutti_i_dati_annuali])
    # Calcolo statistiche prelievi
    statistiche_prelievi = {
        'totale_reale_medio_annuo': np.mean(reddito_reale_annuo_tutte_le_run) if reddito_reale_annuo_tutte_le_run.size > 0 else 0.0
    }
    
    return {
        "statistiche": statistiche,
        "statistiche_prelievi": statistiche_prelievi,
        "dati_grafici_principali": {
            "nominale": patrimoni_nominali_tutte_le_run,
            "reale": patrimoni_reali_tutte_le_run,
            "reddito_reale_annuo": reddito_reale_annuo_tutte_le_run
        },
        "dati_grafici_avanzati": {
            "dati_mediana": dati_mediana_dettagliati
        }
    } 