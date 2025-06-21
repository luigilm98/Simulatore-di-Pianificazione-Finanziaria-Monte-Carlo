import numpy as np

def valida_parametri(parametri):
    """Valida i parametri della simulazione, solleva ValueError se non validi."""
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
    if not (0 <= parametri['rendimento_medio'] <= 1):
        raise ValueError("Rendimento medio deve essere tra 0 e 1")
    if not (0 <= parametri['volatilita'] <= 1):
        raise ValueError("Volatilità deve essere tra 0 e 1")
    if not (0 <= parametri['inflazione'] <= 1):
        raise ValueError("Inflazione deve essere tra 0 e 1")
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

def _esegui_una_simulazione(parametri, prelievo_annuo_da_usare):
    """
    Esegue una singola traiettoria della simulazione Monte Carlo.
    Restituisce un dizionario contenente tutti gli array dei risultati per questa singola esecuzione.
    """
    # Setup iniziale per una singola run
    eta_iniziale = parametri['eta_iniziale']
    mesi_totali = parametri['anni_totali'] * 12
    num_anni = parametri['anni_totali'] + 1

    # Inizializzazione degli array di output per questa run
    # Array mensili
    patrimoni_run = np.zeros(mesi_totali + 1)
    patrimoni_reali_run = np.zeros(mesi_totali + 1)
    patrimonio_storico = []

    # Array annuali
    dati_annuali = {
        'prelievi_target_nominali': np.zeros(num_anni),
        'prelievi_effettivi_nominali': np.zeros(num_anni),
        'prelievi_effettivi_reali': np.zeros(num_anni),
        'prelievi_da_banca_nominali': np.zeros(num_anni),
        'prelievi_da_etf_nominali': np.zeros(num_anni),
        'vendite_rebalance_nominali': np.zeros(num_anni),
        'fp_liquidato_nominale': np.zeros(num_anni),
        'pensioni_pubbliche_nominali': np.zeros(num_anni),
        'pensioni_pubbliche_reali': np.zeros(num_anni),
        'rendite_fp_nominali': np.zeros(num_anni),
        'rendite_fp_reali': np.zeros(num_anni),
        'saldo_banca_nominale': np.zeros(num_anni),
        'saldo_etf_nominale': np.zeros(num_anni),
        'saldo_banca_reale': np.zeros(num_anni),
        'saldo_etf_reale': np.zeros(num_anni),
        'saldo_fp_nominale': np.zeros(num_anni),
        'saldo_fp_reale': np.zeros(num_anni),
        'reddito_totale_reale': np.zeros(num_anni)
    }

    # Stato della simulazione
    patrimonio_banca = parametri['capitale_iniziale']
    patrimonio_etf = parametri['etf_iniziale']
    etf_cost_basis = parametri['etf_iniziale']
    patrimonio_fp = 0
    patrimonio_fp_inizio_anno = 0
    contributi_fp_anno_corrente = 0
    fp_convertito_in_rendita = False
    rendita_mensile_nominale_tassata_fp = 0
    rendita_annua_reale_fp = 0
    indice_prezzi_inizio_rendita_fp = 1.0
    allocazione_etf_inizio_glidepath = -1.0
    
    prelievo_annuo_corrente = 0
    indice_prezzi_ultimo_prelievo = 1.0
    
    indice_prezzi_inizio_pensione = 1.0
    indice_prezzi = 1.0
    
    patrimoni_run[0] = patrimonio_banca + patrimonio_etf
    patrimoni_reali_run[0] = patrimoni_run[0]
    patrimonio_storico.append(patrimoni_run[0])
    patrimonio_negativo = False
    
    if parametri['attiva_fondo_pensione']:
        patrimonio_fp_inizio_anno = patrimonio_fp
    
    inizio_prelievo_mesi = parametri['anni_inizio_prelievo'] * 12 + 1
    
    for mese in range(1, mesi_totali + 1):
        anno_corrente = (mese - 1) // 12
        eta_attuale = eta_iniziale + anno_corrente
        
        if mese < inizio_prelievo_mesi:
            patrimonio_banca += (parametri['contributo_mensile_banca'] * indice_prezzi)
            investimento_indicizzato_etf = (parametri['contributo_mensile_etf'] * indice_prezzi)
            patrimonio_etf += investimento_indicizzato_etf
            etf_cost_basis += investimento_indicizzato_etf
        
        if parametri['attiva_fondo_pensione'] and eta_attuale < parametri['eta_ritiro_fp']:
            contributo_mensile_fp_indicizzato = (parametri['contributo_annuo_fp'] / 12) * indice_prezzi
            patrimonio_fp += contributo_mensile_fp_indicizzato
            contributi_fp_anno_corrente += contributo_mensile_fp_indicizzato

        entrate_passive_mensili = 0
        if anno_corrente >= parametri['inizio_pensione_anni']:
            if anno_corrente == parametri['inizio_pensione_anni'] and (mese-1) % 12 == 0:
                indice_prezzi_inizio_pensione = indice_prezzi
            
            pensione_mensile = (parametri['pensione_pubblica_annua'] / 12) * (indice_prezzi / indice_prezzi_inizio_pensione)
            patrimonio_banca += pensione_mensile
            entrate_passive_mensili += pensione_mensile

        if fp_convertito_in_rendita:
            rendita_mensile_indicizzata = rendita_mensile_nominale_tassata_fp * (indice_prezzi / indice_prezzi_inizio_rendita_fp)
            patrimonio_banca += rendita_mensile_indicizzata
            entrate_passive_mensili += rendita_mensile_indicizzata

        prelievo_mensile = 0
        if mese >= inizio_prelievo_mesi:
            is_primo_mese_prelievo = (mese == inizio_prelievo_mesi)
            is_inizio_anno_fiscale = ((mese - inizio_prelievo_mesi) % 12 == 0)

            if is_primo_mese_prelievo or is_inizio_anno_fiscale:
                patrimonio_attuale = patrimonio_banca + patrimonio_etf
                
                if patrimonio_attuale <= 0:
                    prelievo_annuo_corrente = 0
                else:
                    inflazione_annua_stimata = (indice_prezzi / indice_prezzi_ultimo_prelievo) if indice_prezzi_ultimo_prelievo > 0 else 1.0
                    
                    if is_primo_mese_prelievo:
                        if parametri['strategia_prelievo'] == 'FISSO':
                            prelievo_annuo_corrente = prelievo_annuo_da_usare * indice_prezzi
                        else:
                            prelievo_annuo_corrente = patrimonio_attuale * parametri['percentuale_regola_4']
                    else:
                        if parametri['strategia_prelievo'] == 'FISSO':
                            prelievo_annuo_corrente *= inflazione_annua_stimata
                        elif parametri['strategia_prelievo'] == 'REGOLA_4_PERCENTO':
                            prelievo_annuo_corrente = patrimonio_attuale * parametri['percentuale_regola_4']
                        elif parametri['strategia_prelievo'] == 'GUARDRAIL':
                            prelievo_annuo_corrente *= inflazione_annua_stimata
                            
                            soglia_superiore = parametri['percentuale_regola_4'] * (1 + parametri['banda_guardrail'])
                            soglia_inferiore = parametri['percentuale_regola_4'] * (1 - parametri['banda_guardrail'])
                            tasso_prelievo_attuale = prelievo_annuo_corrente / patrimonio_attuale if patrimonio_attuale > 0 else float('inf')
                            
                            if tasso_prelievo_attuale > soglia_superiore:
                                prelievo_annuo_corrente *= (1 - 0.10)
                            elif tasso_prelievo_attuale < soglia_inferiore:
                                prelievo_annuo_corrente *= (1 + 0.10)
                    
                dati_annuali['prelievi_target_nominali'][anno_corrente] = prelievo_annuo_corrente
                indice_prezzi_ultimo_prelievo = indice_prezzi

            prelievo_mensile = prelievo_annuo_corrente / 12
        
        ricavo_netto_etf_mese = 0
        if prelievo_mensile > 0:
            fabbisogno_liquidita = max(0, prelievo_mensile - patrimonio_banca)
            if fabbisogno_liquidita > 0 and patrimonio_etf > 0:
                cost_basis_ratio = etf_cost_basis / patrimonio_etf if patrimonio_etf > 0 else 1.0
                plusvalenza_ratio = 1.0 - cost_basis_ratio
                tax_rate = parametri['tassazione_capital_gain']
                denominator = 1.0 - (plusvalenza_ratio * tax_rate)
                
                importo_lordo_da_vendere = float('inf')
                if denominator > 1e-9:
                    importo_lordo_da_vendere = fabbisogno_liquidita / denominator
                
                importo_venduto_etf = min(importo_lordo_da_vendere, patrimonio_etf)

                if importo_venduto_etf > 0:
                    costo_proporzionale = (importo_venduto_etf / patrimonio_etf) * etf_cost_basis
                    plusvalenza = importo_venduto_etf - costo_proporzionale
                    tasse_pagate = max(0, plusvalenza * tax_rate)
                    ricavo_netto_etf_mese = importo_venduto_etf - tasse_pagate

                    patrimonio_banca += ricavo_netto_etf_mese
                    patrimonio_etf -= importo_venduto_etf
                    etf_cost_basis -= costo_proporzionale

        prelievo_effettivo_mensile = 0
        if prelievo_mensile > 0:
            prelievo_effettivo_mensile = min(patrimonio_banca, prelievo_mensile)
            patrimonio_banca -= prelievo_effettivo_mensile
        
        if prelievo_effettivo_mensile > 0:
            dati_annuali['prelievi_effettivi_nominali'][anno_corrente] += prelievo_effettivo_mensile
            
            quota_da_etf = min(prelievo_effettivo_mensile, ricavo_netto_etf_mese)
            if quota_da_etf > 0:
                dati_annuali['prelievi_da_etf_nominali'][anno_corrente] += quota_da_etf
            
            quota_da_banca = prelievo_effettivo_mensile - quota_da_etf
            if quota_da_banca > 0:
                dati_annuali['prelievi_da_banca_nominali'][anno_corrente] += quota_da_banca

        dati_annuali['prelievi_effettivi_reali'][anno_corrente] = dati_annuali['prelievi_effettivi_nominali'][anno_corrente] / indice_prezzi

        rendimento_mensile = np.random.normal(parametri['rendimento_medio']/12, parametri['volatilita']/np.sqrt(12))
        patrimonio_etf *= (1 + rendimento_mensile)
        
        if patrimonio_etf > 0:
            patrimonio_etf -= parametri['costo_fisso_etf_mensile']

        if parametri['attiva_fondo_pensione'] and not fp_convertito_in_rendita:
            rendimento_fp = np.random.normal(parametri['rendimento_medio_fp']/12, parametri['volatilita_fp']/np.sqrt(12))
            patrimonio_fp *= (1 + rendimento_fp)

        tasso_inflazione_mensile = np.random.normal(parametri['inflazione']/12, 0.005)
        indice_prezzi *= (1 + tasso_inflazione_mensile)

        if mese % 12 == 0:
            if anno_corrente >= parametri['inizio_pensione_anni']:
                dati_annuali['pensioni_pubbliche_reali'][anno_corrente] = parametri['pensione_pubblica_annua']
                dati_annuali['pensioni_pubbliche_nominali'][anno_corrente] = parametri['pensione_pubblica_annua'] * (indice_prezzi / indice_prezzi_inizio_pensione)
            
            if fp_convertito_in_rendita and eta_attuale >= parametri['eta_ritiro_fp']:
                 dati_annuali['rendite_fp_reali'][anno_corrente] = rendita_annua_reale_fp
                 dati_annuali['rendite_fp_nominali'][anno_corrente] = rendita_annua_reale_fp * indice_prezzi

            dati_annuali['reddito_totale_reale'][anno_corrente] = (
                dati_annuali['prelievi_effettivi_reali'][anno_corrente] +
                dati_annuali['pensioni_pubbliche_reali'][anno_corrente] +
                dati_annuali['rendite_fp_reali'][anno_corrente]
            )

            if patrimonio_banca > 5000: patrimonio_banca -= parametri['imposta_bollo_conto']
            if patrimonio_etf > 0:
                patrimonio_etf -= patrimonio_etf * parametri['imposta_bollo_titoli']
                patrimonio_etf -= patrimonio_etf * parametri['ter_etf']
            
            if parametri['attiva_fondo_pensione'] and not fp_convertito_in_rendita:
                patrimonio_fp_post_costi = patrimonio_fp * (1 - parametri['ter_fp'])
                
                rendimento_maturato_anno = patrimonio_fp_post_costi - patrimonio_fp_inizio_anno - contributi_fp_anno_corrente
                
                if rendimento_maturato_anno > 0:
                    tassa_su_rendimento = rendimento_maturato_anno * parametri['tassazione_rendimenti_fp']
                    patrimonio_fp = patrimonio_fp_post_costi - tassa_su_rendimento
                else:
                    patrimonio_fp = patrimonio_fp_post_costi

                patrimonio_fp_inizio_anno = patrimonio_fp
                contributi_fp_anno_corrente = 0
                
                if eta_attuale >= parametri['eta_ritiro_fp'] and patrimonio_fp > 0:
                    capitale_ritirato = patrimonio_fp * parametri['percentuale_capitale_fp']
                    dati_annuali['fp_liquidato_nominale'][anno_corrente] = capitale_ritirato
                    patrimonio_banca += capitale_ritirato * (1 - parametri['aliquota_finale_fp'])
                    
                    capitale_per_rendita = patrimonio_fp * (1 - parametri['percentuale_capitale_fp'])
                    if parametri['durata_rendita_fp_anni'] > 0 and capitale_per_rendita > 0:
                        tasso_interesse_annuo_fp = parametri['rendimento_medio_fp'] - parametri['ter_fp']
                        n_anni_rendita = parametri['durata_rendita_fp_anni']
                        
                        rendita_annua_nominale_iniziale = 0
                        if tasso_interesse_annuo_fp > 0:
                            fattore_rendita = tasso_interesse_annuo_fp / (1 - (1 + tasso_interesse_annuo_fp) ** -n_anni_rendita)
                            rendita_annua_nominale_iniziale = capitale_per_rendita * fattore_rendita
                        else:
                            rendita_annua_nominale_iniziale = capitale_per_rendita / n_anni_rendita

                        rendita_annua_reale_fp = rendita_annua_nominale_iniziale / indice_prezzi
                        rendita_mensile_nominale_tassata_fp = (rendita_annua_nominale_iniziale / 12) * (1 - parametri['aliquota_finale_fp'])
                        indice_prezzi_inizio_rendita_fp = indice_prezzi
                    
                    patrimonio_fp = 0
                    fp_convertito_in_rendita = True

            if parametri['attiva_glidepath']:
                patrimonio_investibile = patrimonio_banca + patrimonio_etf
                if patrimonio_investibile > 0:
                    current_etf_alloc = patrimonio_etf / patrimonio_investibile
                    target_alloc_etf = -1.0
                    
                    if anno_corrente >= parametri['inizio_glidepath_anni'] and anno_corrente <= parametri['fine_glidepath_anni']:
                        if allocazione_etf_inizio_glidepath < 0:
                            allocazione_etf_inizio_glidepath = current_etf_alloc
                        
                        durata = float(parametri['fine_glidepath_anni'] - parametri['inizio_glidepath_anni'])
                        progresso = (anno_corrente - parametri['inizio_glidepath_anni']) / durata if durata > 0 else 1.0
                        target_alloc_etf = allocazione_etf_inizio_glidepath * (1 - progresso) + parametri['allocazione_etf_finale'] * progresso
                    
                    elif anno_corrente > parametri['fine_glidepath_anni']:
                        target_alloc_etf = parametri['allocazione_etf_finale']

                    if target_alloc_etf >= 0:
                        rebalance_amount = (patrimonio_investibile * target_alloc_etf) - patrimonio_etf
                        if rebalance_amount < -1:
                            importo_da_vendere = abs(rebalance_amount)
                            dati_annuali['vendite_rebalance_nominali'][anno_corrente] += importo_da_vendere
                            costo_proporzionale = (importo_da_vendere / patrimonio_etf) * etf_cost_basis if patrimonio_etf > 0 else 0
                            plusvalenza = importo_da_vendere - costo_proporzionale
                            tasse_da_pagare = max(0, plusvalenza * parametri['tassazione_capital_gain'])
                            
                            patrimonio_etf -= (importo_da_vendere)
                            etf_cost_basis -= costo_proporzionale
                            patrimonio_banca += (importo_da_vendere - tasse_da_pagare)

                        elif rebalance_amount > 1:
                            importo_da_comprare = min(rebalance_amount, patrimonio_banca)
                            patrimonio_banca -= importo_da_comprare
                            patrimonio_etf += importo_da_comprare
                            etf_cost_basis += importo_da_comprare
                            
            dati_annuali['saldo_banca_nominale'][anno_corrente] = patrimonio_banca
            dati_annuali['saldo_etf_nominale'][anno_corrente] = patrimonio_etf
            dati_annuali['saldo_banca_reale'][anno_corrente] = patrimonio_banca / indice_prezzi
            dati_annuali['saldo_etf_reale'][anno_corrente] = patrimonio_etf / indice_prezzi
            dati_annuali['saldo_fp_nominale'][anno_corrente] = patrimonio_fp
            dati_annuali['saldo_fp_reale'][anno_corrente] = patrimonio_fp / indice_prezzi

            if parametri['attiva_fondo_pensione']:
                patrimonio_fp_inizio_anno = patrimonio_fp

        patrimoni_run[mese] = max(0, patrimonio_banca + patrimonio_etf + patrimonio_fp)
        patrimoni_reali_run[mese] = max(0, patrimoni_run[mese] / indice_prezzi)
        patrimonio_storico.append(patrimoni_run[mese])
        
        if (patrimonio_banca + patrimonio_etf) <= 0 and mese >= inizio_prelievo_mesi:
            patrimonio_negativo = True

    # Calcoli finali per la singola run
    drawdown = 0
    sharpe_ratio = 0
    with np.errstate(divide='ignore', invalid='ignore'):
        patrimonio_np = np.array(patrimonio_storico)
        picchi = np.maximum.accumulate(patrimonio_np)
        if np.any(picchi > 0):
            drawdown_values = (patrimonio_np - picchi) / picchi
            drawdown = np.min(drawdown_values)
        
        returns = patrimoni_run[1:] / patrimoni_run[:-1] - 1
        finite_returns = returns[np.isfinite(returns)]
        if finite_returns.size > 1 and np.std(finite_returns) > 0:
            risk_free_month = (1 + 0.02) ** (1/12) - 1
            sharpe_ratio = (np.mean(finite_returns) - risk_free_month) * np.sqrt(12) / np.std(finite_returns)
            
    return {
        "patrimoni_mensili": patrimoni_run,
        "patrimoni_reali_mensili": patrimoni_reali_run,
        "dati_annuali": dati_annuali,
        "drawdown": drawdown,
        "sharpe_ratio": sharpe_ratio,
        "fallimento": patrimonio_negativo
    }

def run_full_simulation(parametri):
    """
    Esegue la simulazione completa e restituisce tutti i dati aggregati e calcolati.
    """
    # 1. SETUP INIZIALE
    valida_parametri(parametri)

    n_sim = parametri['n_simulazioni']
    mesi_totali = parametri['anni_totali'] * 12
    num_anni = parametri['anni_totali'] + 1
    
    patrimoni = np.zeros((n_sim, mesi_totali + 1))
    patrimoni_reali = np.zeros((n_sim, mesi_totali + 1))
    drawdowns = np.zeros(n_sim)
    sharpe_ratios = np.zeros(n_sim)
    fallimenti = 0
    
    prelievi_annuali_reali_agg = np.zeros((n_sim, num_anni))
    rendite_fp_annuali_reali_agg = np.zeros((n_sim, num_anni))
    pensioni_annuali_reali_agg = np.zeros((n_sim, num_anni))
    saldi_fp_reali_agg = np.zeros((n_sim, num_anni))
    redditi_totali_reali_agg = np.zeros((n_sim, num_anni))

    prelievo_annuo_da_usare = parametri['prelievo_annuo']
    calcolo_sostenibile_attivo = parametri['strategia_prelievo'] == 'FISSO' and parametri['prelievo_annuo'] == 0

    if calcolo_sostenibile_attivo:
        prelievo_annuo_da_usare = 0

    # 2. ESECUZIONE SIMULAZIONI
    for sim in range(n_sim):
        risultati_run = _esegui_una_simulazione(parametri, prelievo_annuo_da_usare)
        
        patrimoni[sim, :] = risultati_run['patrimoni_mensili']
        patrimoni_reali[sim, :] = risultati_run['patrimoni_reali_mensili']
        drawdowns[sim] = risultati_run['drawdown']
        sharpe_ratios[sim] = risultati_run['sharpe_ratio']
        if risultati_run['fallimento']:
            fallimenti += 1
        
        prelievi_annuali_reali_agg[sim, :] = risultati_run['dati_annuali']['prelievi_effettivi_reali']
        rendite_fp_annuali_reali_agg[sim, :] = risultati_run['dati_annuali']['rendite_fp_reali']
        pensioni_annuali_reali_agg[sim, :] = risultati_run['dati_annuali']['pensioni_pubbliche_reali']
        saldi_fp_reali_agg[sim, :] = risultati_run['dati_annuali']['saldo_fp_reale']
        redditi_totali_reali_agg[sim, :] = risultati_run['dati_annuali']['reddito_totale_reale']

    # 3. CALCOLO STATISTICHE E SCENARIO MEDIANO
    prob_fallimento = fallimenti / n_sim
    patrimoni_finale_validi = patrimoni[:, -1]
    patrimoni_reali_finale_validi = patrimoni_reali[:, -1]
    
    dati_per_foglio_prelievi = {}

    if calcolo_sostenibile_attivo:
        anni_accumulo = parametri['anni_inizio_prelievo']
        anni_prelievo = parametri['anni_totali'] - anni_accumulo
        patrimonio_reale_finale_mediano_no_prelievi = np.median(patrimoni_reali_finale_validi)

        prelievo_annuo_calcolato = 0
        if anni_prelievo > 0 and patrimonio_reale_finale_mediano_no_prelievi > 0:
            alloc_etf_ritiro = parametri['allocazione_etf_finale']
            rend_reale_etf = parametri['rendimento_medio'] - parametri['ter_etf'] - parametri['inflazione']
            rend_reale_cash = 0 - parametri['inflazione']
            tasso_reale_sconto = (alloc_etf_ritiro * rend_reale_etf) + ((1 - alloc_etf_ritiro) * rend_reale_cash)

            if tasso_reale_sconto != 0:
                try:
                    fattore_rendita = tasso_reale_sconto / (1 - (1 + tasso_reale_sconto) ** -anni_prelievo)
                    prelievo_annuo_calcolato = patrimonio_reale_finale_mediano_no_prelievi * fattore_rendita
                except (OverflowError, ZeroDivisionError):
                    prelievo_annuo_calcolato = patrimonio_reale_finale_mediano_no_prelievi / anni_prelievo
            else:
                prelievo_annuo_calcolato = patrimonio_reale_finale_mediano_no_prelievi / anni_prelievo
        
        risultati_mediana_con_prelievo = _esegui_una_simulazione(parametri, prelievo_annuo_calcolato)
        dati_per_foglio_prelievi = risultati_mediana_con_prelievo['dati_annuali']

    else:
        if len(patrimoni_reali_finale_validi) > 0:
            median_sim_index = np.argmin(np.abs(patrimoni_reali[:, -1] - np.median(patrimoni_reali_finale_validi)))
        else:
            median_sim_index = 0
        risultati_mediana = _esegui_una_simulazione(parametri, prelievo_annuo_da_usare)
        dati_per_foglio_prelievi = risultati_mediana['dati_annuali']
    
    # --- Calcolo Statistiche Prelievi ---
    anno_inizio_prelievo = parametri['anni_inizio_prelievo']
    anno_inizio_pensione = parametri['inizio_pensione_anni']
    anno_inizio_rendita_fp = parametri['eta_ritiro_fp'] - parametri['eta_iniziale']

    anno_inizio_reddito_pensione = min(anno_inizio_prelievo, anno_inizio_pensione)
    if parametri['attiva_fondo_pensione'] and anno_inizio_rendita_fp < parametri['anni_totali']:
        anno_inizio_reddito_pensione = min(anno_inizio_reddito_pensione, anno_inizio_rendita_fp)

    totali_reali_agg = prelievi_annuali_reali_agg + pensioni_annuali_reali_agg + rendite_fp_annuali_reali_agg
    totali_reali_attivi = totali_reali_agg[:, anno_inizio_reddito_pensione:]
    totali_reali_validi = totali_reali_attivi[totali_reali_attivi > 1e-6]

    prelievi_reali_attivi = prelievi_annuali_reali_agg[:, anno_inizio_prelievo:]
    prelievi_reali_validi = prelievi_reali_attivi[prelievi_reali_attivi > 1e-6]

    rendite_fp_reali_attive = rendite_fp_annuali_reali_agg[:, anno_inizio_rendita_fp:]
    rendite_fp_validi = rendite_fp_reali_attive[rendite_fp_reali_attive > 1e-6]

    statistiche_prelievi = {
        'prelievo_reale_medio': np.mean(prelievi_reali_validi) if prelievi_reali_validi.size > 0 else 0,
        'pensione_pubblica_reale_annua': parametri['pensione_pubblica_annua'],
        'rendita_fp_reale_media': np.mean(rendite_fp_validi) if rendite_fp_validi.size > 0 else 0,
        'totale_reale_medio_annuo': np.mean(totali_reali_validi) if totali_reali_validi.size > 0 else 0,
    }

    # 4. PREPARAZIONE OUTPUT
    statistiche_finali = {
        'patrimonio_iniziale': parametri['capitale_iniziale'] + parametri['etf_iniziale'],
        'patrimonio_finale_mediano_nominale': np.median(patrimoni_finale_validi),
        'patrimonio_finale_top_10_nominale': np.percentile(patrimoni_finale_validi, 90),
        'patrimonio_finale_peggior_10_nominale': np.percentile(patrimoni_finale_validi, 10),
        'patrimonio_finale_mediano_reale': np.median(patrimoni_reali_finale_validi),
        'drawdown_massimo_peggiore': np.min(drawdowns) if drawdowns.size > 0 else 0,
        'sharpe_ratio_medio': np.mean(sharpe_ratios[np.isfinite(sharpe_ratios)]),
        'probabilita_fallimento': prob_fallimento,
        'patrimoni_reali_finali': patrimoni_reali_finale_validi,
        'successo_per_anno': np.sum(patrimoni_reali[:, ::12] > 1, axis=0) / n_sim if n_sim > 0 else np.zeros(parametri['anni_totali'] + 1),
    }

    return {
        "statistiche": statistiche_finali,
        "statistiche_prelievi": statistiche_prelievi,
        "dati_grafici_principali": {
            "nominale": patrimoni,
            "reale": patrimoni_reali,
            "reddito_reale_annuo": redditi_totali_reali_agg
        },
        "dati_grafici_avanzati": {
            "dati_mediana": dati_per_foglio_prelievi
        }
    } 