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
        if not (0 <= parametri['aliquota_finale_fp'] <= 1):
            raise ValueError("Aliquota finale FP deve essere tra 0 e 1 (es. 0.15 per 15%)")

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
        'tasse_rebalance_nominali': np.zeros(num_anni),
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
    totale_contributi_versati_nominale = 0
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
            contrib_banca = (parametri['contributo_mensile_banca'] * indice_prezzi)
            contrib_etf = (parametri['contributo_mensile_etf'] * indice_prezzi)
            patrimonio_banca += contrib_banca
            patrimonio_etf += contrib_etf
            etf_cost_basis += contrib_etf
            totale_contributi_versati_nominale += contrib_banca + contrib_etf
        
        if parametri['attiva_fondo_pensione'] and eta_attuale < parametri['eta_ritiro_fp']:
            contributo_mensile_fp_indicizzato = (parametri['contributo_annuo_fp'] / 12) * indice_prezzi
            patrimonio_fp += contributo_mensile_fp_indicizzato
            contributi_fp_anno_corrente += contributo_mensile_fp_indicizzato
            totale_contributi_versati_nominale += contributo_mensile_fp_indicizzato

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
            patrimonio_etf -= patrimonio_etf * (parametri['ter_etf'] / 12)
            patrimonio_etf -= parametri['costo_fisso_etf_mensile']

        if parametri['attiva_fondo_pensione'] and not fp_convertito_in_rendita:
            rendimento_fp = np.random.normal(parametri['rendimento_medio_fp']/12, parametri['volatilita_fp']/np.sqrt(12))
            patrimonio_fp *= (1 + rendimento_fp)
            if patrimonio_fp > 0:
                patrimonio_fp -= patrimonio_fp * (parametri['ter_fp'] / 12)

        tasso_inflazione_mensile = np.random.normal(parametri['inflazione']/12, 0.005)
        indice_prezzi *= (1 + tasso_inflazione_mensile)

        if mese % 12 == 0:
            if anno_corrente >= parametri['inizio_pensione_anni']:
                dati_annuali['pensioni_pubbliche_reali'][anno_corrente] = parametri['pensione_pubblica_annua']
                dati_annuali['pensioni_pubbliche_nominali'][anno_corrente] = parametri['pensione_pubblica_annua'] * (indice_prezzi / indice_prezzi_inizio_pensione)
            
            if fp_convertito_in_rendita and eta_attuale >= parametri['eta_ritiro_fp']:
                 dati_annuali['rendite_fp_reali'][anno_corrente] = rendita_annua_reale_fp
                 dati_annuali['rendite_fp_nominali'][anno_corrente] = rendita_annua_reale_fp * (indice_prezzi / indice_prezzi_inizio_rendita_fp)

            dati_annuali['reddito_totale_reale'][anno_corrente] = (
                dati_annuali['prelievi_effettivi_reali'][anno_corrente] +
                dati_annuali['pensioni_pubbliche_reali'][anno_corrente] +
                dati_annuali['rendite_fp_reali'][anno_corrente]
            )

            # --- Logica Glidepath (fine anno) ---
            if parametri['attiva_glidepath'] and anno_corrente >= parametri['inizio_glidepath_anni']:
                patrimonio_investibile = patrimonio_banca + patrimonio_etf
                if patrimonio_investibile > 0:
                    current_etf_alloc = patrimonio_etf / patrimonio_investibile
                    
                    if allocazione_etf_inizio_glidepath < 0:
                         allocazione_etf_inizio_glidepath = current_etf_alloc

                    durata = parametri['fine_glidepath_anni'] - parametri['inizio_glidepath_anni']
                    progresso = (anno_corrente - parametri['inizio_glidepath_anni']) / durata if durata > 0 else 1.0
                    progresso = min(progresso, 1.0)

                    target_alloc_etf = allocazione_etf_inizio_glidepath * (1 - progresso) + parametri['allocazione_etf_finale'] * progresso
                    
                    rebalance_amount = (patrimonio_investibile * target_alloc_etf) - patrimonio_etf

                    if rebalance_amount < -1: # Vendi ETF
                        importo_da_vendere = abs(rebalance_amount)
                        dati_annuali['vendite_rebalance_nominali'][anno_corrente] += importo_da_vendere
                        
                        costo_proporzionale = (importo_da_vendere / patrimonio_etf) * etf_cost_basis if patrimonio_etf > 0 else 0
                        plusvalenza = importo_da_vendere - costo_proporzionale
                        tasse_da_pagare = max(0, plusvalenza * parametri['tassazione_capital_gain'])
                        
                        dati_annuali['tasse_rebalance_nominali'][anno_corrente] += tasse_da_pagare

                        patrimonio_etf -= (importo_da_vendere)
                        etf_cost_basis -= costo_proporzionale
                        patrimonio_banca += (importo_da_vendere - tasse_da_pagare)

                    elif rebalance_amount > 1: # Compra ETF
                        importo_da_comprare = min(rebalance_amount, patrimonio_banca)
                        patrimonio_banca -= importo_da_comprare
                        patrimonio_etf += importo_da_comprare
                        etf_cost_basis += importo_da_comprare
            
            # Liquidazione / Conversione in Rendita Fondo Pensione (fine anno)
            if parametri['attiva_fondo_pensione'] and eta_attuale == parametri['eta_ritiro_fp'] - 1 and not fp_convertito_in_rendita:
                montante_fp = patrimonio_fp
                tasse_fp = montante_fp * parametri['aliquota_finale_fp']
                montante_netto_fp = montante_fp - tasse_fp
                dati_annuali['fp_liquidato_nominale'][anno_corrente] = montante_netto_fp
                
                if parametri['tipo_liquidazione_fp'] == 'Rendita':
                    fattore_conversione_grezzo = 0.045 # Esempio
                    rendita_annua_lorda = montante_netto_fp * fattore_conversione_grezzo
                    rendita_annua_reale_fp = rendita_annua_lorda / indice_prezzi
                    
                    rendita_mensile_nominale_tassata_fp = rendita_annua_lorda / 12
                    fp_convertito_in_rendita = True
                    indice_prezzi_inizio_rendita_fp = indice_prezzi
                    patrimonio_fp = 0 
                else: # 'Capitale'
                    patrimonio_banca += montante_netto_fp
                    patrimonio_fp = 0
            
            # Salvataggio dati di fine anno
            dati_annuali['saldo_banca_nominale'][anno_corrente] = patrimonio_banca
            dati_annuali['saldo_etf_nominale'][anno_corrente] = patrimonio_etf
            dati_annuali['saldo_banca_reale'][anno_corrente] = patrimonio_banca / indice_prezzi
            dati_annuali['saldo_etf_reale'][anno_corrente] = patrimonio_etf / indice_prezzi
            
            if parametri['attiva_fondo_pensione']:
                dati_annuali['saldo_fp_nominale'][anno_corrente] = patrimonio_fp
                dati_annuali['saldo_fp_reale'][anno_corrente] = patrimonio_fp / indice_prezzi
            
            contributi_fp_anno_corrente = 0
            patrimonio_fp_inizio_anno = patrimonio_fp

        patrimoni_run[mese] = patrimonio_banca + patrimonio_etf + patrimonio_fp
        patrimoni_reali_run[mese] = patrimoni_run[mese] / indice_prezzi
        patrimonio_storico.append(patrimoni_run[mese])
        
        if (patrimonio_banca + patrimonio_etf) <= 0 and mese >= inizio_prelievo_mesi:
            patrimonio_negativo = True

    # Calcoli finali per la singola run
    drawdown = 0
    sharpe_ratio = 0
    if len(patrimonio_storico) > 1:
        rendimenti_mensili_np = np.diff(patrimonio_storico) / patrimonio_storico[:-1]
        rendimento_medio_mensile = np.mean(rendimenti_mensili_np) if len(rendimenti_mensili_np) > 0 else 0
        std_dev_mensile = np.std(rendimenti_mensili_np) if len(rendimenti_mensili_np) > 0 else 0
        sharpe_ratio = (rendimento_medio_mensile / std_dev_mensile) * np.sqrt(12) if std_dev_mensile > 0 else 0
        
        rolling_max = np.maximum.accumulate(patrimonio_storico)
        drawdowns = (rolling_max - patrimonio_storico) / rolling_max
        drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0

    return {
        'patrimoni_mensili': patrimoni_run,
        'patrimoni_reali_mensili': patrimoni_reali_run,
        'dati_annuali': dati_annuali,
        'fallimento': patrimonio_negativo,
        'patrimonio_finale_reale': patrimoni_reali_run[-1],
        'contributi_totali_nominali': totale_contributi_versati_nominale,
        'max_drawdown': drawdown,
        'sharpe_ratio': sharpe_ratio
    }

def esegui_simulazioni(parametri, prelievo_annuo_da_usare):
    valida_parametri(parametri)
    n_sim = parametri['n_simulazioni']
    anni_totali = parametri['anni_totali']
    mesi_totali = anni_totali * 12
    num_anni = anni_totali + 1

    patrimoni = np.zeros((n_sim, mesi_totali + 1))
    patrimoni_reali = np.zeros((n_sim, mesi_totali + 1))
    prelievi_target_nominali_sim = np.zeros((n_sim, num_anni))
    prelievi_effettivi_nominali_sim = np.zeros((n_sim, num_anni))
    prelievi_effettivi_reali_sim = np.zeros((n_sim, num_anni))
    reddito_totale_reale_sim = np.zeros((n_sim, num_anni))
    tasse_rebalance_nominali_sim = np.zeros((n_sim, num_anni))

    fallimenti = np.zeros(n_sim, dtype=bool)
    patrimoni_finali_reali = np.zeros(n_sim)
    prelievi_medi_reali = np.zeros(n_sim)
    contributi_totali_nominali_agg = np.zeros(n_sim)
    max_drawdowns_agg = np.zeros(n_sim)
    sharpe_ratios_agg = np.zeros(n_sim)

    # Dizionario per accumulare tutti i dati annuali da tutte le simulazioni
    tutti_i_dati_annuali = {
        'prelievi_da_banca_nominali': np.zeros((n_sim, num_anni)),
        'prelievi_da_etf_nominali': np.zeros((n_sim, num_anni)),
        'vendite_rebalance_nominali': np.zeros((n_sim, num_anni)),
        'fp_liquidato_nominale': np.zeros((n_sim, num_anni)),
        'pensioni_pubbliche_nominali': np.zeros((n_sim, num_anni)),
        'pensioni_pubbliche_reali': np.zeros((n_sim, num_anni)),
        'rendite_fp_nominali': np.zeros((n_sim, num_anni)),
        'rendite_fp_reali': np.zeros((n_sim, num_anni)),
        'saldo_banca_nominale': np.zeros((n_sim, num_anni)),
        'saldo_etf_nominale': np.zeros((n_sim, num_anni)),
        'saldo_banca_reale': np.zeros((n_sim, num_anni)),
        'saldo_etf_reale': np.zeros((n_sim, num_anni)),
        'saldo_fp_nominale': np.zeros((n_sim, num_anni)),
        'saldo_fp_reale': np.zeros((n_sim, num_anni)),
        'prelievi_effettivi_reali': np.zeros((n_sim, num_anni))
    }

    if parametri['strategia_prelievo'] == 'FISSO':
        prelievo_annuo_da_usare = parametri['prelievo_annuo']

    for sim in range(n_sim):
        risultati_run = _esegui_una_simulazione(parametri, prelievo_annuo_da_usare)
        
        patrimoni[sim, :] = risultati_run['patrimoni_mensili']
        patrimoni_reali[sim, :] = risultati_run['patrimoni_reali_mensili']
        prelievi_target_nominali_sim[sim, :] = risultati_run['dati_annuali']['prelievi_target_nominali']
        prelievi_effettivi_nominali_sim[sim, :] = risultati_run['dati_annuali']['prelievi_effettivi_nominali']
        prelievi_effettivi_reali_sim[sim, :] = risultati_run['dati_annuali']['prelievi_effettivi_reali']
        reddito_totale_reale_sim[sim, :] = risultati_run['dati_annuali']['reddito_totale_reale']
        tasse_rebalance_nominali_sim[sim, :] = risultati_run['dati_annuali']['tasse_rebalance_nominali']
        
        fallimenti[sim] = risultati_run['fallimento']
        patrimoni_finali_reali[sim] = risultati_run['patrimonio_finale_reale']
        contributi_totali_nominali_agg[sim] = risultati_run['contributi_totali_nominali']
        max_drawdowns_agg[sim] = risultati_run['max_drawdown']
        sharpe_ratios_agg[sim] = risultati_run['sharpe_ratio']

        for key in tutti_i_dati_annuali:
             if key in risultati_run['dati_annuali']:
                tutti_i_dati_annuali[key][sim, :] = risultati_run['dati_annuali'][key]

    tasso_fallimento = np.mean(fallimenti)
    scenari_successo_mask = ~fallimenti

    if np.any(scenari_successo_mask):
        patrimoni_finali_reali_validi = patrimoni_finali_reali[scenari_successo_mask]
        
        # Usa reddito_totale_reale_sim per le statistiche sul reddito
        reddito_totale_reale_validi = reddito_totale_reale_sim[scenari_successo_mask]

        anni_prelievo = parametri['anni_totali'] - parametri['anni_inizio_prelievo']
        if anni_prelievo > 0:
            # Calcola il reddito medio annuo nel periodo di decumulo
            reddito_periodo_decumulo = reddito_totale_reale_validi[:, parametri['anni_inizio_prelievo']:]
            redditi_medi_reali_validi = np.mean(reddito_periodo_decumulo, axis=1)
        else:
            redditi_medi_reali_validi = np.array([0])

        statistiche_prelievi = {
            'min': np.min(redditi_medi_reali_validi),
            'p10': np.percentile(redditi_medi_reali_validi, 10),
            'p25': np.percentile(redditi_medi_reali_validi, 25),
            'mediana': np.median(redditi_medi_reali_validi),
            'p75': np.percentile(redditi_medi_reali_validi, 75),
            'p90': np.percentile(redditi_medi_reali_validi, 90),
            'max': np.max(redditi_medi_reali_validi),
            'medio': np.mean(redditi_medi_reali_validi)
        }
        
        stats_patrimonio_reale = {
            'min': np.min(patrimoni_finali_reali_validi),
            'p10': np.percentile(patrimoni_finali_reali_validi, 10),
            'p25': np.percentile(patrimoni_finali_reali_validi, 25),
            'mediana': np.median(patrimoni_finali_reali_validi),
            'p75': np.percentile(patrimoni_finali_reali_validi, 75),
            'p90': np.percentile(patrimoni_finali_reali_validi, 90),
            'max': np.max(patrimoni_finali_reali_validi),
            'medio': np.mean(patrimoni_finali_reali_validi)
        }
    else:
        statistiche_prelievi = {k: 0 for k in ['min', 'p10', 'p25', 'mediana', 'p75', 'p90', 'max', 'medio']}
        stats_patrimonio_reale = {k: 0 for k in ['min', 'p10', 'p25', 'mediana', 'p75', 'p90', 'max', 'medio']}

    dati_mediana_run = {}
    # Aggiungiamo anche i prelievi effettivi reali per la tabella
    tutti_i_dati_annuali['prelievi_effettivi_reali'] = prelievi_effettivi_reali_sim
    for key, data_array in tutti_i_dati_annuali.items():
        dati_mediana_run[key] = np.median(data_array, axis=0)

    prob_successo_nel_tempo = np.mean(patrimoni[:, 1:] > 0, axis=0)
    
    num_worst_cases = max(1, n_sim // 20)
    indici_peggiori = np.argsort(patrimoni_finali_reali)[:num_worst_cases]
    worst_scenarios_data = patrimoni_reali[indici_peggiori, :]
    patrimoni_finali_peggiori = patrimoni_finali_reali[indici_peggiori]

    return {
        'patrimoni_reali': patrimoni_reali,
        'prelievi_target_nominali': prelievi_target_nominali_sim,
        'prelievi_effettivi_nominali': prelievi_effettivi_nominali_sim,
        'prelievi_effettivi_reali': prelievi_effettivi_reali_sim,
        'reddito_totale_reale': reddito_totale_reale_sim,
        'tasse_rebalance_nominali': tasse_rebalance_nominali_sim,
        'statistiche': {
            'tasso_fallimento': tasso_fallimento,
            'percentili_patrimonio_reale': stats_patrimonio_reale,
            'statistiche_prelievi': statistiche_prelievi,
            'contributi_totali_mediani': np.median(contributi_totali_nominali_agg),
            'max_drawdown_mediano': np.median(max_drawdowns_agg),
            'sharpe_ratio_mediano': np.median(sharpe_ratios_agg),
        },
        "dati_grafici_avanzati": {
            "dati_mediana": dati_mediana_run,
            "prob_successo_nel_tempo": prob_successo_nel_tempo,
            "worst_scenarios": {
                "traiettorie": worst_scenarios_data,
                "patrimoni_finali": patrimoni_finali_peggiori
            }
        }
    }