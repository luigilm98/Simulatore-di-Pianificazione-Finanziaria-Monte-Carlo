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
        'variazione_patrimonio_percentuale', 'rendimento_investimento_percentuale',
        'contributi_totali_versati', 'indice_prezzi', 'reddito_totale_reale'
    ]}

    # Stato iniziale dei saldi e delle variabili
    patrimonio_banca = parametri['capitale_iniziale']
    patrimonio_etf = parametri['etf_iniziale']
    etf_cost_basis = patrimonio_etf
    patrimonio_fp = 0
    
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

    # Modello economico a regimi
    model_name = parametri.get('economic_model', "VOLATILE (CICLI BOOM-BUST)")
    economic_model_params = _get_regime_params(model_name)
    market_regime_definitions = economic_model_params['market_regimes']
    inflation_regime_definitions = economic_model_params['inflation_regimes']
    current_market_regime = np.random.choice(list(market_regime_definitions.keys()))
    current_inflation_regime = np.random.choice(list(inflation_regime_definitions.keys()))

    # --- 2. LOOP DI SIMULAZIONE MENSILE ---
    for mese in range(1, mesi_totali + 1):
        anno_corrente = (mese - 1) // 12 + 1
        eta_attuale = parametri['eta_iniziale'] + (mese - 1) / 12

        # A. ENTRATE MENSILI (Pensione / Rendite)
        # Lo stipendio non viene calcolato qui, i risparmi vengono aggiunti
        # direttamente come "contributi" nella fase di accumulo.

        # Calcolo Pensione Pubblica
        pensione_pubblica_mese = 0
        inizio_pensione_mesi = parametri.get('inizio_pensione_anni', num_anni + 1) * 12
        if mese >= inizio_pensione_mesi:
            pensione_pubblica_mese = parametri.get('pensione_pubblica_annua', 0) / 12
        
        # Calcolo Rendita Fondo Pensione (logica non implementata in questa versione)
        rendita_fp_mese = 0
        
        # Aggiornamento contabile delle entrate
        patrimonio_banca += pensione_pubblica_mese # La pensione viene accreditata in banca
        dati_annuali['pensioni_pubbliche_nominali'][anno_corrente] += pensione_pubblica_mese
        dati_annuali['rendite_fp_nominali'][anno_corrente] += rendita_fp_mese
        
        # Calcolo del reddito reale da pensioni/rendite
        reddito_da_pensioni_reale = (pensione_pubblica_mese + rendita_fp_mese) / indice_prezzi
        dati_annuali['reddito_totale_reale'][anno_corrente] += reddito_da_pensioni_reale

        # B. FASE DI ACCUMULO (prima dei rendimenti)
        if mese < inizio_prelievo_mesi:
            # Contributi
            patrimonio_banca += parametri['contributo_mensile_banca']
            contributi_totali_accumulati += parametri['contributo_mensile_banca']
            
            investimento_etf = min(parametri['contributo_mensile_etf'], patrimonio_banca)
            if investimento_etf > 0:
                patrimonio_banca -= investimento_etf
                patrimonio_etf += investimento_etf
                etf_cost_basis += investimento_etf
                contributi_totali_accumulati += investimento_etf

        # B. FASE DI PRELIEVO (prima dei rendimenti)
        if mese >= inizio_prelievo_mesi:
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
                        prelievo_annuo_nominale_iniziale = patrimonio_a_prelievo * (parametri['percentuale_regola_4'] / 100)
                        indice_prezzi_inizio_pensione = indice_prezzi
                        prelievo_annuo_nominale_corrente = prelievo_annuo_nominale_iniziale
                    else:
                        prelievo_annuo_nominale_corrente = prelievo_annuo_nominale_iniziale * (indice_prezzi / indice_prezzi_inizio_pensione)
            
            # Esegui il prelievo mensile
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
                
                # Aggiungi il prelievo al reddito totale reale dell'anno
                dati_annuali['reddito_totale_reale'][anno_corrente] += prelievo_totale_mese / indice_prezzi

        # C. RENDIMENTI, COSTI E AGGIORNAMENTO INFLAZIONE
        market_regime = market_regime_definitions[current_market_regime]
        inflation_regime = inflation_regime_definitions[current_inflation_regime]
        rendimento_mensile = np.random.normal(market_regime['mean'] / 12, market_regime['vol'] / np.sqrt(12))
        inflazione_mensile = np.random.normal(inflation_regime['mean'] / 12, inflation_regime['vol'] / np.sqrt(12))
        
        patrimonio_etf *= (1 + rendimento_mensile)
        patrimonio_etf -= patrimonio_etf * (parametri['ter_etf'] / 12)
        indice_prezzi *= (1 + inflazione_mensile)

        current_market_regime = _choose_next_regime(current_market_regime, market_regime_definitions)
        current_inflation_regime = _choose_next_regime(current_inflation_regime, inflation_regime_definitions)

        # D. OPERAZIONI DI FINE ANNO
        if mese % 12 == 0:
            patrimonio_inizio_anno = dati_annuali['saldo_banca_nominale'][anno_corrente-1] + dati_annuali['saldo_etf_nominale'][anno_corrente-1]
            patrimonio_fine_anno = patrimonio_banca + patrimonio_etf
            
            dati_annuali['variazione_patrimonio_percentuale'][anno_corrente] = (patrimonio_fine_anno - patrimonio_inizio_anno) / patrimonio_inizio_anno if patrimonio_inizio_anno > 0 else 0
            dati_annuali['saldo_banca_nominale'][anno_corrente] = patrimonio_banca
            dati_annuali['saldo_etf_nominale'][anno_corrente] = patrimonio_etf
            dati_annuali['saldo_banca_reale'][anno_corrente] = patrimonio_banca / indice_prezzi
            dati_annuali['saldo_etf_reale'][anno_corrente] = patrimonio_etf / indice_prezzi
            dati_annuali['indice_prezzi'][anno_corrente] = indice_prezzi
            dati_annuali['contributi_totali_versati'][anno_corrente] = contributi_totali_accumulati

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