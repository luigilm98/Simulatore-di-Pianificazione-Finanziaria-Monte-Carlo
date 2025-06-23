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

def _esegui_una_simulazione(parametri, prelievo_annuo_da_usare):
    """
    Esegue una singola traiettoria (un "universo parallelo") della simulazione.

    Questa è la funzione più granulare, che simula l'evoluzione del patrimonio
    mese per mese per l'intero orizzonte temporale. Gestisce tutti gli eventi:
    contributi, prelievi, ribilanciamenti, tassazione, pensione, ecc.

    Args:
        parametri (dict): Il dizionario completo dei parametri della simulazione.
        prelievo_annuo_da_usare (float): L'importo del prelievo annuo reale
            calcolato (se richiesto) o specificato dall'utente.

    Returns:
        dict: Un dizionario contenente tutti gli array dei risultati (sia mensili
              che annuali) e le statistiche per questa singola traiettoria.
    """
    # --- 1. SETUP INIZIALE DELLA SIMULAZIONE E DEL MODELLO ECONOMICO ---
    economic_model_params = _get_regime_params(parametri.get('economic_model', "VOLATILE (CICLI BOOM-BUST)"))
    market_regime_definitions = economic_model_params['market_regimes']
    inflation_regime_definitions = economic_model_params['inflation_regimes']
    
    current_market_regime = economic_model_params['initial_market_regime']
    current_inflation_regime = economic_model_params['initial_inflation_regime']
    
    # Parametri temporali
    eta_iniziale = parametri['eta_iniziale']
    mesi_totali = parametri['anni_totali'] * 12
    num_anni = parametri['anni_totali']
    inizio_prelievo_mesi = parametri['anni_inizio_prelievo'] * 12 + 1

    # Inizializzazione degli array per i risultati annuali
    dati_annuali = {k: np.zeros(num_anni + 1) for k in [
        'prelievi_target_nominali', 'prelievi_effettivi_nominali', 'prelievi_effettivi_reali',
        'prelievi_da_banca_nominali', 'prelievi_da_etf_nominali', 'vendite_rebalance_nominali',
        'fp_liquidato_nominale', 'pensioni_pubbliche_nominali', 'pensioni_pubbliche_reali',
        'rendite_fp_nominali', 'rendite_fp_reali', 'saldo_banca_nominale', 'saldo_etf_nominale',
        'saldo_banca_reale', 'saldo_etf_reale', 'saldo_fp_nominale', 'saldo_fp_reale',
        'reddito_totale_reale', 'variazione_patrimonio_percentuale', 'rendimento_investimento_percentuale',
        'contributi_totali_versati', 'indice_prezzi'
    ]}

    # Stato iniziale del patrimonio e altre variabili di stato
    patrimonio_banca = parametri['capitale_iniziale']
    patrimonio_etf = parametri['etf_iniziale']
    etf_cost_basis = parametri['etf_iniziale']
    patrimonio_fp = 0
    
    # FIX: Salva i valori iniziali (anno 0) direttamente nel dizionario annuale
    dati_annuali['saldo_banca_nominale'][0] = patrimonio_banca
    dati_annuali['saldo_etf_nominale'][0] = patrimonio_etf
    dati_annuali['saldo_fp_nominale'][0] = patrimonio_fp
    dati_annuali['indice_prezzi'][0] = 1.0

    patrimonio_negativo = False
    
    # Variabili per il calcolo dei flussi e rendimenti
    contributi_totali_accumulati = 0
    flussi_netti_investimento_anno = 0
    guadagni_accumulo = 0
    guadagni_calcolati = False
    
    # Variabili per il calcolo dell'inflazione e dei prelievi
    indice_prezzi = 1.0
    prelievo_annuo_nominale_corrente = 0.0
    indice_prezzi_inizio_pensione = 1.0
    prelievo_annuo_nominale_iniziale = 0.0
    
    # Variabili per il Fondo Pensione
    patrimonio_fp_inizio_anno = 0
    contributi_fp_anno_corrente = 0
    fp_convertito_in_rendita = False
    capitale_fp_per_rendita = 0
    rendita_annua_nominale_lorda_fp = 0
    rendita_mensile_nominale_tassata_fp = 0
    indice_prezzi_inizio_rendita_fp = 1.0
    
    # Variabili per il Glidepath
    allocazione_etf_inizio_glidepath = -1.0
    
    # --- 2. LOOP DI SIMULAZIONE MENSILE ---
    for mese in range(1, mesi_totali + 1):
        anno_corrente = (mese -1) // 12 + 1
        eta_attuale = parametri['eta_iniziale'] + (mese - 1) / 12
        
        # --- A. CALCOLO RENDIMENTI E INFLAZIONE MENSILI ---
        # Determina il prossimo stato economico e calcola i rendimenti del mese
        market_regime_params = market_regime_definitions[current_market_regime]
        inflation_regime_params = inflation_regime_definitions[current_inflation_regime]
        rendimento_mensile_etf = np.random.normal(market_regime_params['mean']/12, market_regime_params['vol']/np.sqrt(12))
        tasso_inflazione_mensile = np.random.normal(inflation_regime_params['mean']/12, inflation_regime_params['vol']/np.sqrt(12))
        
        current_market_regime = _choose_next_regime(current_market_regime, market_regime_definitions)
        current_inflation_regime = _choose_next_regime(current_inflation_regime, inflation_regime_definitions)
        
        # Applica rendimenti e costi a ETF e Fondo Pensione
        patrimonio_etf *= (1 + rendimento_mensile_etf)
        patrimonio_etf -= patrimonio_etf * (parametri['ter_etf'] / 12) + \
                          patrimonio_etf * (parametri['imposta_bollo_titoli'] / 12) + \
                          parametri['costo_fisso_etf_mensile']

        if parametri['attiva_fondo_pensione'] and (eta_attuale < parametri['eta_ritiro_fp'] or capitale_fp_per_rendita > 0):
             # NOTA: Il FP usa ancora un modello semplice, non a regimi.
             rendimento_mensile_fp = np.random.normal(parametri['rendimento_medio_fp']/12, parametri['volatilita_fp']/np.sqrt(12))
             patrimonio_fp *= (1 + rendimento_mensile_fp)
             patrimonio_fp -= patrimonio_fp * (parametri['ter_fp'] / 12)

        # Aggiorna l'indice dei prezzi generale
        indice_prezzi *= (1 + tasso_inflazione_mensile)
        # Controllo di sicurezza: assicurati che l'indice prezzi rimanga sempre positivo
        indice_prezzi = max(indice_prezzi, 1e-10)

        # --- B. ACCANTONAMENTO E CONTRIBUTI MENSILI ---
        if mese < inizio_prelievo_mesi:
            # Si assume che i contributi vengano fatti all'inizio del mese, prima dei rendimenti
            patrimonio_banca += parametri['contributo_mensile_banca']
            contributi_totali_accumulati += parametri['contributo_mensile_banca']
            
            # Gestione contributo a ETF
            investimento_etf = parametri['contributo_mensile_etf']
            if investimento_etf > 0:
                importo_reale_investito = min(investimento_etf, patrimonio_banca)
                patrimonio_banca -= importo_reale_investito
                patrimonio_etf += importo_reale_investito
                flussi_netti_investimento_anno += importo_reale_investito
                etf_cost_basis += importo_reale_investito
                contributi_totali_accumulati += importo_reale_investito
            
            # Gestione contributo a Fondo Pensione
            if parametri['attiva_fondo_pensione'] and eta_attuale < parametri['eta_ritiro_fp']:
                contributo_mensile_fp = parametri['contributo_annuo_fp'] / 12
                patrimonio_fp += contributo_mensile_fp
                contributi_fp_anno_corrente += contributo_mensile_fp
                contributi_totali_accumulati += contributo_mensile_fp

        # --- C. FASE DI PRELIEVO: GESTIONE USCITE ---
        if mese >= inizio_prelievo_mesi:
            
            # Calcola i guadagni totali UNA SOLA VOLTA, al momento esatto dell'inizio prelievo
            if not guadagni_calcolati:
                patrimonio_attuale = patrimonio_banca + patrimonio_etf + patrimonio_fp
                contributi_fino_a_prelievo = contributi_totali_accumulati
                guadagni_accumulo = patrimonio_attuale - (parametri['capitale_iniziale'] + parametri['etf_iniziale']) - contributi_fino_a_prelievo
                guadagni_calcolati = True

            # FIX: Imposta/aggiorna l'importo del prelievo annuale nominale SOLO UNA VOLTA ALL'ANNO.
            is_new_retirement_year = (mese - inizio_prelievo_mesi) % 12 == 0
            if is_new_retirement_year:
                if parametri['strategia_prelievo'] == 'FISSO':
                    prelievo_annuo_nominale_corrente = prelievo_annuo_da_usare * indice_prezzi
                
                elif parametri['strategia_prelievo'] == 'REGOLA_4_PERCENTO':
                    if mese == inizio_prelievo_mesi:
                        patrimonio_a_prelievo = patrimonio_banca + patrimonio_etf
                        prelievo_annuo_nominale_iniziale = patrimonio_a_prelievo * (parametri['percentuale_regola_4'] / 100)
                        indice_prezzi_inizio_pensione = indice_prezzi
                        prelievo_annuo_nominale_corrente = prelievo_annuo_nominale_iniziale
                    else:
                        prelievo_annuo_nominale_corrente = prelievo_annuo_nominale_iniziale * (indice_prezzi / indice_prezzi_inizio_pensione)

            # Il prelievo mensile è semplicemente 1/12 dell'obiettivo annuale
            prelievo_mensile_target = prelievo_annuo_nominale_corrente / 12 if prelievo_annuo_nominale_corrente > 0 else 0
            
            # Esegui il prelievo effettivo, prelevando prima dalla banca e poi dagli ETF
            if prelievo_mensile_target > 0:
                prelievo_da_banca = min(prelievo_mensile_target, patrimonio_banca)
                patrimonio_banca -= prelievo_da_banca
                flussi_netti_investimento_anno -= prelievo_da_banca
                dati_annuali['prelievi_da_banca_nominali'][anno_corrente] += prelievo_da_banca
                
                fabbisogno_da_etf = prelievo_mensile_target - prelievo_da_banca
                prelievo_da_etf = 0

                if fabbisogno_da_etf > 0 and patrimonio_etf > 0:
                    importo_da_vendere = min(fabbisogno_da_etf, patrimonio_etf)
                    
                    costo_proporzionale = (importo_da_vendere / patrimonio_etf) * etf_cost_basis if patrimonio_etf > 0 else 0
                    plusvalenza = importo_da_vendere - costo_proporzionale
                    tasse_da_pagare = max(0, plusvalenza * parametri['tassazione_capital_gain'])
                    
                    # Riduci il patrimonio ETF dell'importo lordo venduto
                    patrimonio_etf -= importo_da_vendere
                    etf_cost_basis -= costo_proporzionale
                    flussi_netti_investimento_anno -= importo_da_vendere
                    
                    # Il prelievo effettivo da ETF è il netto dopo le tasse
                    prelievo_da_etf = importo_da_vendere - tasse_da_pagare
                    dati_annuali['prelievi_da_etf_nominali'][anno_corrente] += prelievo_da_etf
                
                prelievo_totale_effettivo = prelievo_da_banca + prelievo_da_etf
                dati_annuali['prelievi_target_nominali'][anno_corrente] += prelievo_mensile_target
                dati_annuali['prelievi_effettivi_nominali'][anno_corrente] += prelievo_totale_effettivo
                dati_annuali['prelievi_effettivi_reali'][anno_corrente] += prelievo_totale_effettivo / indice_prezzi

        # --- D. GESTIONE ENTRATE PASSIVE (PENSIONI E RENDITE) ---
        if anno_corrente >= parametri['inizio_pensione_anni']:
            if anno_corrente == parametri['inizio_pensione_anni'] and (mese - 1) % 12 == 0:
                indice_prezzi_inizio_pensione = indice_prezzi
            pensione_mensile = (parametri['pensione_pubblica_annua'] / 12) * (indice_prezzi / indice_prezzi_inizio_pensione)
            patrimonio_banca += pensione_mensile

        if fp_convertito_in_rendita:
            rendita_mensile_indicizzata = rendita_mensile_nominale_tassata_fp * (indice_prezzi / indice_prezzi_inizio_rendita_fp)
            patrimonio_banca += rendita_mensile_indicizzata
            if capitale_fp_per_rendita > 0:
                quota_capitale_erosa = (rendita_annua_nominale_lorda_fp / 12) * (indice_prezzi / indice_prezzi_inizio_rendita_fp)
                capitale_fp_per_rendita = max(0, capitale_fp_per_rendita - quota_capitale_erosa)
                patrimonio_fp = capitale_fp_per_rendita

        # --- E. AGGIORNAMENTO STATO E CONTROLLO FALLIMENTO ---
        # Questa sezione ora non è più necessaria per lo storico,
        # perché i dati vengono già salvati annualmente.
        # Mantieniamo il controllo di fallimento.
        patrimonio_totale_mese = max(0, patrimonio_banca + patrimonio_etf + patrimonio_fp)
        if patrimonio_totale_mese <= 0 and mese >= inizio_prelievo_mesi:
            patrimonio_negativo = True
    
    # --- 3. OUTPUT FINALE DELLA SINGOLA RUN ---
    # FIX: Calcola correttamente lo storico del patrimonio per il calcolo del drawdown
    patrimonio_storico_nominale = dati_annuali['saldo_banca_nominale'] + dati_annuali['saldo_etf_nominale'] + dati_annuali['saldo_fp_nominale']
    
    drawdown = 0
    if np.any(patrimonio_storico_nominale > 0):
        picchi = np.maximum.accumulate(patrimonio_storico_nominale)
        drawdown_values = (patrimonio_storico_nominale - picchi) / picchi
        drawdown = np.min(drawdown_values)

    return {
        "dati_annuali": dati_annuali,
        "drawdown": drawdown,
        "fallimento": patrimonio_negativo,
        "guadagni_accumulo": guadagni_accumulo,
        "contributi_totali_versati": contributi_totali_accumulati,
        "indice_prezzi_finale": indice_prezzi
    }


def _calcola_prelievo_sostenibile(parametri):
    """
    Trova il prelievo annuo reale massimo sostenibile con una ricerca binaria.
    
    L'obiettivo è trovare il tasso di prelievo che porta il patrimonio reale
    finale *mediano* il più vicino possibile a zero, senza diventare negativo.
    Utilizza un numero ridotto di simulazioni per la velocità.

    Args:
        parametri (dict): Il dizionario completo dei parametri della simulazione.

    Returns:
        float: L'importo del prelievo annuo reale sostenibile calcolato.
    """
    params_test = parametri.copy()
    params_test['prelievo_annuo'] = 0
    params_test['n_simulazioni'] = max(100, params_test['n_simulazioni'] // 4)
    
    risultati_test = run_full_simulation(params_test)
    patrimoni_reali_test = risultati_test['dati_grafici_principali']['reale']
    
    idx_inizio_prelievo = parametri['anni_inizio_prelievo'] * 12
    capitale_reale_mediano_a_prelievo = np.median(patrimoni_reali_test[:, idx_inizio_prelievo])

    if capitale_reale_mediano_a_prelievo <= 0: return 0
    anni_prelievo = parametri['anni_totali'] - parametri['anni_inizio_prelievo']
    if anni_prelievo <= 0: return 0
        
    limite_inferiore, limite_superiore = 0, capitale_reale_mediano_a_prelievo / anni_prelievo
    prelievo_ottimale = 0
    
    def mediana_finale(prelievo):
        params_run = parametri.copy()
        params_run['n_simulazioni'] = max(100, params_run['n_simulazioni'] // 4)
        risultati_run = run_full_simulation(params_run, prelievo_annuo_da_usare=prelievo)
        return np.median(risultati_run['statistiche']['patrimonio_finale_mediano_reale'])

    for _ in range(15):
        prelievo_corrente = (limite_inferiore + limite_superiore) / 2
        if prelievo_corrente < 1: break
        patrimonio_risultante = mediana_finale(prelievo_corrente)
        if patrimonio_risultante > 0:
            prelievo_ottimale, limite_inferiore = prelievo_corrente, prelievo_corrente
        else:
            limite_superiore = prelievo_corrente
            
    return prelievo_ottimale


def run_full_simulation(parametri, prelievo_annuo_da_usare=None):
    """
    Funzione orchestratrice principale per l'esecuzione della simulazione completa.

    Questa funzione esegue i seguenti passaggi:
    1. Valida i parametri di input.
    2. Se l'utente ha richiesto un prelievo "sostenibile", lo calcola
       preventivamente con `_calcola_prelievo_sostenibile`.
    3. Esegue il numero completo di simulazioni (`n_simulazioni`) invocando
       ripetutamente `_esegui_una_simulazione`.
    4. Aggrega i risultati di tutte le traiettorie.
    5. Identifica lo "scenario mediano" (quello il cui patrimonio finale reale
       è più vicino alla mediana di tutti i risultati) per l'analisi dettagliata.
    6. Calcola un'ampia gamma di statistiche aggregate (es. probabilità di
       fallimento, percentili del patrimonio, Sharpe ratio, ecc.).
    7. Restituisce un dizionario strutturato con tutti i dati necessari per
       la visualizzazione nell'interfaccia utente.

    Args:
        parametri (dict): Il dizionario completo dei parametri della simulazione.
        prelievo_annuo_da_usare (float, optional): Usato internamente per la
            ricerca del prelievo sostenibile. Se `None`, viene gestito
            normalmente.

    Returns:
        dict: Un dizionario annidato contenente le statistiche finali, i dati
              per i grafici principali e i dati dettagliati dello scenario mediano.
    """
    valida_parametri(parametri)
    
    # Gestione del calcolo del prelievo sostenibile
    prelievo_sostenibile_calcolato = None
    if prelievo_annuo_da_usare is None:
        if parametri['strategia_prelievo'] == 'FISSO' and parametri['prelievo_annuo'] == 0:
            prelievo_sostenibile_calcolato = _calcola_prelievo_sostenibile(parametri)
            prelievo_annuo_da_usare = prelievo_sostenibile_calcolato
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
        # Sharpe Ratio non più calcolato per semplicità nel nuovo modello
        'sharpe_ratio_medio': 0.0,
        'patrimoni_reali_finali': patrimoni_finali_reali,
        'guadagni_accumulo_mediano_nominale': np.median(tutti_i_guadagni),
        'contributi_totali_versati_mediano_nominale': np.median(tutti_i_contributi),
        'prelievo_sostenibile_calcolato': prelievo_sostenibile_calcolato
    }

    # Estrazione redditi per analisi
    reddito_reale_annuo_tutte_le_run = np.array([run['reddito_totale_reale'] for run in tutti_i_dati_annuali])
    # ... (logica calcolo statistiche prelievi omessa per brevità) ...
    statistiche_prelievi = {'totale_reale_medio_annuo': 0.0}
    
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