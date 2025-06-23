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
        'prelievi_target_nominali': np.zeros(num_anni + 1),
        'prelievi_effettivi_nominali': np.zeros(num_anni + 1),
        'prelievi_effettivi_reali': np.zeros(num_anni + 1),
        'prelievi_da_banca_nominali': np.zeros(num_anni + 1),
        'prelievi_da_etf_nominali': np.zeros(num_anni + 1),
        'vendite_rebalance_nominali': np.zeros(num_anni + 1),
        'fp_liquidato_nominale': np.zeros(num_anni + 1),
        'pensioni_pubbliche_nominali': np.zeros(num_anni + 1),
        'pensioni_pubbliche_reali': np.zeros(num_anni + 1),
        'rendite_fp_nominali': np.zeros(num_anni + 1),
        'rendite_fp_reali': np.zeros(num_anni + 1),
        'saldo_banca_nominale': np.zeros(num_anni + 1),
        'saldo_etf_nominale': np.zeros(num_anni + 1),
        'saldo_banca_reale': np.zeros(num_anni + 1),
        'saldo_etf_reale': np.zeros(num_anni + 1),
        'saldo_fp_nominale': np.zeros(num_anni + 1),
        'saldo_fp_reale': np.zeros(num_anni + 1),
        'reddito_totale_reale': np.zeros(num_anni + 1),
        'variazione_patrimonio_percentuale': np.zeros(num_anni + 1),
        'rendimento_investimento_percentuale': np.zeros(num_anni + 1)
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
    
    guadagni_accumulo = 0
    guadagni_calcolati = False
    guadagni_investimento_anno_nominale = 0

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
    
    # Stato specifico per la gestione della rendita del FP
    capitale_fp_per_rendita = 0
    rendita_annua_nominale_lorda_fp = 0
    
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
        
        if mese == inizio_prelievo_mesi and not guadagni_calcolati:
            patrimonio_a_inizio_prelievo = patrimonio_banca + patrimonio_etf + patrimonio_fp
            patrimonio_iniziale_totale = parametri['capitale_iniziale'] + parametri['etf_iniziale']
            guadagni_accumulo = patrimonio_a_inizio_prelievo - patrimonio_iniziale_totale - totale_contributi_versati_nominale
            guadagni_calcolati = True

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
            # Riduci il capitale rimanente nel FP
            if capitale_fp_per_rendita > 0:
                quota_capitale_erosa = (rendita_annua_nominale_lorda_fp / 12) * (indice_prezzi / indice_prezzi_inizio_rendita_fp)
                capitale_fp_per_rendita = max(0, capitale_fp_per_rendita - quota_capitale_erosa)
                patrimonio_fp = capitale_fp_per_rendita # Aggiorna il saldo principale del FP

        prelievo_mensile = 0
        if mese >= inizio_prelievo_mesi:
            is_primo_mese_prelievo = (mese == inizio_prelievo_mesi)
            is_inizio_anno_fiscale = ((mese - inizio_prelievo_mesi) % 12 == 0)

            if is_primo_mese_prelievo or is_inizio_anno_fiscale:
                patrimonio_attuale = patrimonio_banca + patrimonio_etf
                
                # FIX: Esclude la liquidazione del capitale FP dell'anno precedente dalla base di calcolo
                # per le strategie di prelievo basate su percentuali, per evitare picchi anomali.
                if anno_corrente > 0:
                    lump_sum_anno_precedente = dati_annuali['fp_liquidato_nominale'][anno_corrente - 1]
                    if lump_sum_anno_precedente > 0:
                        patrimonio_attuale -= lump_sum_anno_precedente

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

        # Il calcolo del valore reale viene spostato a fine anno per coerenza
        # dati_annuali['prelievi_effettivi_reali'][anno_corrente] = dati_annuali['prelievi_effettivi_nominali'][anno_corrente] / indice_prezzi

        # Calcolo dei costi mensili da applicare
        costo_ter_mensile = parametri['ter_etf'] / 12
        costo_bollo_mensile = parametri['imposta_bollo_titoli'] / 12

        # Calcolo rendimento e applicazione costi
        rendimento_mensile = np.random.normal(parametri['rendimento_medio']/12, parametri['volatilita']/np.sqrt(12))
        
        # Applica evento di mercato estremo se abilitato
        evento_moltiplicatore = _genera_evento_mercato_estremo(parametri.get('eventi_mercato_estremi', 'DISABILITATI'))
        rendimento_mensile = (1 + rendimento_mensile) * evento_moltiplicatore - 1
        
        # 1. Applica il rendimento lordo
        guadagno_etf_mese = patrimonio_etf * rendimento_mensile
        guadagni_investimento_anno_nominale += guadagno_etf_mese
        patrimonio_etf *= (1 + rendimento_mensile)
        
        # 2. Applica i costi sugli ETF
        patrimonio_etf -= patrimonio_etf * costo_ter_mensile
        patrimonio_etf -= patrimonio_etf * costo_bollo_mensile
        patrimonio_etf -= parametri['costo_fisso_etf_mensile']

        # 3. Applica rendimento e costi al Fondo Pensione (se attivo e prima del ritiro completo)
        if parametri['attiva_fondo_pensione'] and (eta_attuale < parametri['eta_ritiro_fp'] or capitale_fp_per_rendita > 0):
             rendimento_mensile_fp = np.random.normal(parametri['rendimento_medio_fp']/12, parametri['volatilita_fp']/np.sqrt(12))
             
             guadagno_fp_mese = 0
             # Se la rendita è attiva, il rendimento si applica solo sul capitale residuo
             if fp_convertito_in_rendita:
                 guadagno_fp_mese = patrimonio_fp * rendimento_mensile_fp
                 patrimonio_fp *= (1 + rendimento_mensile_fp)
             else: # Altrimenti sul totale accumulato
                 guadagno_fp_mese = patrimonio_fp * rendimento_mensile_fp
                 patrimonio_fp *= (1 + rendimento_mensile_fp)

             guadagni_investimento_anno_nominale += guadagno_fp_mese
             patrimonio_fp -= patrimonio_fp * (parametri['ter_fp'] / 12)

        tasso_inflazione_mensile = np.random.normal(parametri['inflazione']/12, 0.005)
        indice_prezzi *= (1 + tasso_inflazione_mensile)

        if mese % 12 == 0:
            # Calcolo dei valori reali a fine anno
            dati_annuali['prelievi_effettivi_reali'][anno_corrente] = dati_annuali['prelievi_effettivi_nominali'][anno_corrente] / indice_prezzi

            if anno_corrente >= parametri['inizio_pensione_anni']:
                dati_annuali['pensioni_pubbliche_reali'][anno_corrente] = parametri['pensione_pubblica_annua']
                dati_annuali['pensioni_pubbliche_nominali'][anno_corrente] = parametri['pensione_pubblica_annua'] * (indice_prezzi / indice_prezzi_inizio_pensione)
            
            if fp_convertito_in_rendita and eta_attuale >= parametri['eta_ritiro_fp']:
                 dati_annuali['rendite_fp_reali'][anno_corrente] = rendita_annua_reale_fp
                 dati_annuali['rendite_fp_nominali'][anno_corrente] = rendita_annua_reale_fp * indice_prezzi

            reddito_reale_anno = (
                dati_annuali['prelievi_effettivi_reali'][anno_corrente] +
                dati_annuali['pensioni_pubbliche_reali'][anno_corrente] +
                dati_annuali['rendite_fp_reali'][anno_corrente]
            )
            dati_annuali['reddito_totale_reale'][anno_corrente] = reddito_reale_anno

            if patrimonio_banca > 5000: patrimonio_banca -= parametri['imposta_bollo_conto']
            
            # Eseguiamo il ribilanciamento PRIMA della liquidazione del FP,
            # per evitare che la liquidazione venga immediatamente reinvestita.
            strategia_ribilanciamento = parametri.get('strategia_ribilanciamento', 'GLIDEPATH')

            if strategia_ribilanciamento != 'NESSUNO':
                target_alloc_etf = -1.0 

                if strategia_ribilanciamento == 'GLIDEPATH':
                    if anno_corrente >= parametri['inizio_glidepath_anni'] and anno_corrente <= parametri['fine_glidepath_anni']:
                        patrimonio_investibile_check = patrimonio_banca + patrimonio_etf
                        if patrimonio_investibile_check > 0 and allocazione_etf_inizio_glidepath < 0:
                             allocazione_etf_inizio_glidepath = patrimonio_etf / patrimonio_investibile_check
                        
                        durata = float(parametri['fine_glidepath_anni'] - parametri['inizio_glidepath_anni'])
                        progresso = (anno_corrente - parametri['inizio_glidepath_anni']) / durata if durata > 0 else 1.0
                        
                        # Usa l'allocazione di partenza salvata per un calcolo più stabile
                        alloc_partenza = allocazione_etf_inizio_glidepath if allocazione_etf_inizio_glidepath >= 0 else (patrimonio_etf / (patrimonio_banca + patrimonio_etf) if (patrimonio_banca + patrimonio_etf) > 0 else 0)
                        target_alloc_etf = alloc_partenza * (1 - progresso) + parametri['allocazione_etf_finale'] * progresso
                    
                    elif anno_corrente > parametri['fine_glidepath_anni']:
                        target_alloc_etf = parametri['allocazione_etf_finale']

                elif strategia_ribilanciamento == 'ANNUALE_FISSO':
                    target_alloc_etf = parametri.get('allocazione_etf_fissa', 0.60)

                # Logica comune di esecuzione del ribilanciamento
                if target_alloc_etf >= 0:
                    patrimonio_investibile = patrimonio_banca + patrimonio_etf
                    if patrimonio_investibile > 0:
                        rebalance_amount = (patrimonio_investibile * target_alloc_etf) - patrimonio_etf
                        if rebalance_amount < -1: # Vendi per ribilanciare
                            importo_da_vendere = abs(rebalance_amount)
                            dati_annuali['vendite_rebalance_nominali'][anno_corrente] += importo_da_vendere
                            costo_proporzionale = (importo_da_vendere / patrimonio_etf) * etf_cost_basis if patrimonio_etf > 0 else 0
                            plusvalenza = importo_da_vendere - costo_proporzionale
                            tasse_da_pagare = max(0, plusvalenza * parametri['tassazione_capital_gain'])
                            
                            patrimonio_etf -= importo_da_vendere
                            etf_cost_basis -= costo_proporzionale
                            patrimonio_banca += (importo_da_vendere - tasse_da_pagare)

                        elif rebalance_amount > 1: # Compra per ribilanciare
                            importo_da_comprare = min(rebalance_amount, patrimonio_banca)
                            patrimonio_banca -= importo_da_comprare
                            patrimonio_etf += importo_da_comprare
                            etf_cost_basis += importo_da_comprare

            if parametri['attiva_fondo_pensione'] and not fp_convertito_in_rendita:
                # La gestione del rendimento tassato sul FP è complessa e richiede il confronto con l'anno precedente
                # quindi la lasciamo nel blocco annuale.
                patrimonio_fp_post_costi = patrimonio_fp # I costi TER sono già stati applicati mensilmente
                
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
                    
                    capitale_fp_per_rendita = patrimonio_fp * (1 - parametri['percentuale_capitale_fp'])
                    if parametri['durata_rendita_fp_anni'] > 0 and capitale_fp_per_rendita > 0:
                        tasso_interesse_annuo_fp = parametri['rendimento_medio_fp'] - parametri['ter_fp']
                        n_anni_rendita = parametri['durata_rendita_fp_anni']
                        
                        rendita_annua_nominale_iniziale = 0
                        if tasso_interesse_annuo_fp > 0:
                            fattore_rendita = tasso_interesse_annuo_fp / (1 - (1 + tasso_interesse_annuo_fp) ** -n_anni_rendita)
                            rendita_annua_nominale_iniziale = capitale_fp_per_rendita * fattore_rendita
                        else:
                            rendita_annua_nominale_iniziale = capitale_fp_per_rendita / n_anni_rendita

                        # Salva la rendita lorda per l'erosione del capitale
                        rendita_annua_nominale_lorda_fp = rendita_annua_nominale_iniziale

                        rendita_annua_reale_fp = rendita_annua_nominale_iniziale / indice_prezzi
                        rendita_mensile_nominale_tassata_fp = (rendita_annua_nominale_iniziale / 12) * (1 - parametri['aliquota_finale_fp'])
                        indice_prezzi_inizio_rendita_fp = indice_prezzi
                    
                    # ERRORE CORRETTO: il patrimonio FP non si azzera, ma diventa il capitale per la rendita.
                    patrimonio_fp = capitale_fp_per_rendita
                    fp_convertito_in_rendita = True

            dati_annuali['saldo_banca_nominale'][anno_corrente] = patrimonio_banca
            dati_annuali['saldo_etf_nominale'][anno_corrente] = patrimonio_etf
            dati_annuali['saldo_banca_reale'][anno_corrente] = patrimonio_banca / indice_prezzi
            dati_annuali['saldo_etf_reale'][anno_corrente] = patrimonio_etf / indice_prezzi
            dati_annuali['saldo_fp_nominale'][anno_corrente] = patrimonio_fp
            dati_annuali['saldo_fp_reale'][anno_corrente] = patrimonio_fp / indice_prezzi

            # Calcolo della variazione percentuale annua del patrimonio
            patrimonio_totale_anno_corrente = patrimonio_banca + patrimonio_etf + patrimonio_fp
            
            patrimonio_totale_anno_precedente = 0
            if anno_corrente == 0:
                patrimonio_totale_anno_precedente = parametri['capitale_iniziale'] + parametri['etf_iniziale']
            else:
                patrimonio_totale_anno_precedente = (
                    dati_annuali['saldo_banca_nominale'][anno_corrente - 1] +
                    dati_annuali['saldo_etf_nominale'][anno_corrente - 1] +
                    dati_annuali['saldo_fp_nominale'][anno_corrente - 1]
                )
                if patrimonio_totale_anno_precedente > 0:
                    variazione = (patrimonio_totale_anno_corrente - patrimonio_totale_anno_precedente) / patrimonio_totale_anno_precedente
                    dati_annuali['variazione_patrimonio_percentuale'][anno_corrente] = variazione

            # NUOVO: Rendimento puro da investimento
            rendimento_investimento = guadagni_investimento_anno_nominale / patrimonio_totale_anno_precedente
            dati_annuali['rendimento_investimento_percentuale'][anno_corrente] = rendimento_investimento

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
        patrimoni_np = np.array(patrimonio_storico)
        picchi = np.maximum.accumulate(patrimoni_np)
        if np.any(picchi > 0):
            drawdown_values = (patrimoni_np - picchi) / picchi
            drawdown = np.min(drawdown_values)
        
        returns = patrimoni_run[1:] / patrimoni_run[:-1] - 1
        finite_returns = returns[np.isfinite(returns)]
        if finite_returns.size > 1 and np.std(finite_returns) > 0:
            risk_free_month = (1 + 0.02) ** (1/12) - 1 # Assumiamo un tasso risk-free del 2% annuo
            sharpe_ratio = (np.mean(finite_returns) - risk_free_month) * np.sqrt(12) / np.std(finite_returns)
            if not np.isfinite(sharpe_ratio):
                sharpe_ratio = 0
            
    # Popoliamo i dati dell'ultimo anno che altrimenti rimarrebbero a zero
    final_year_index = parametri['anni_totali']
    dati_annuali['saldo_banca_nominale'][final_year_index] = patrimonio_banca
    dati_annuali['saldo_etf_nominale'][final_year_index] = patrimonio_etf
    dati_annuali['saldo_fp_nominale'][final_year_index] = patrimonio_fp
    dati_annuali['saldo_banca_reale'][final_year_index] = patrimonio_banca / indice_prezzi
    dati_annuali['saldo_etf_reale'][final_year_index] = patrimonio_etf / indice_prezzi
    dati_annuali['saldo_fp_reale'][final_year_index] = patrimonio_fp / indice_prezzi
            
    return {
        "patrimoni_run": patrimoni_run,
        "patrimoni_reali_run": patrimoni_reali_run,
        "reddito_annuo_reale": dati_annuali['reddito_totale_reale'][:parametri['anni_totali']],
        "dati_annuali": dati_annuali,
        "drawdown": drawdown,
        "sharpe_ratio": sharpe_ratio,
        "fallimento": patrimonio_negativo,
        "totale_contributi_versati_nominale": totale_contributi_versati_nominale,
        "guadagni_accumulo": guadagni_accumulo
    }

def _calcola_prelievo_sostenibile(parametri):
    """
    Trova il prelievo annuo reale massimo sostenibile che porta il patrimonio
    reale finale mediano il più vicino possibile a zero, senza diventare negativo.
    Utilizza un approccio di ricerca binaria.
    """
    
    # Esegui una simulazione senza prelievi per stimare il capitale al momento del ritiro
    params_test = parametri.copy()
    params_test['prelievo_annuo'] = 0
    params_test['n_simulazioni'] = max(100, params_test['n_simulazioni'] // 4) # Riduci le simulazioni per velocità
    
    # Usa una versione "light" della simulazione per la stima iniziale
    _, patrimoni_reali, _, _, _ = _esegui_simulazioni_principali(params_test, 0)
    
    # Stima del capitale reale mediano all'inizio dei prelievi
    idx_inizio_prelievo = parametri['anni_inizio_prelievo'] * 12
    capitale_reale_mediano_a_prelievo = np.median(patrimoni_reali[:, idx_inizio_prelievo])

    if capitale_reale_mediano_a_prelievo <= 0:
        return 0

    # Definisci i limiti della ricerca binaria
    anni_di_prelievo = parametri['anni_totali'] - parametri['anni_inizio_prelievo']
    if anni_di_prelievo <= 0:
        return 0
        
    limite_inferiore = 0
    # Un limite superiore "aggressivo": consumare tutto il capitale in modo lineare
    limite_superiore = capitale_reale_mediano_a_prelievo / anni_di_prelievo 

    prelievo_ottimale = 0
    
    # Wrapper per la funzione obiettivo
    def mediana_finale(prelievo_test):
        """Esegue la simulazione e restituisce il patrimonio reale finale mediano."""
        _, patrimoni_reali_test, _, _, _ = _esegui_simulazioni_principali(params_test, prelievo_test)
        patrimonio_finale_mediano = np.median(patrimoni_reali_test[:, -1])
        return patrimonio_finale_mediano

    # Ciclo di ricerca binaria per trovare il prelievo ottimale
    for _ in range(15): # 15 iterazioni sono sufficienti per una buona convergenza
        prelievo_corrente = (limite_inferiore + limite_superiore) / 2
        if prelievo_corrente < 1: # Evita loop con valori troppo piccoli
            break

        patrimonio_risultante = mediana_finale(prelievo_corrente)

        if patrimonio_risultante > 0:
            # Possiamo permetterci di prelevare di più
            prelievo_ottimale = prelievo_corrente
            limite_inferiore = prelievo_corrente
        else:
            # Dobbiamo prelevare di meno
            limite_superiore = prelievo_corrente
            
    return prelievo_ottimale

def _esegui_simulazioni_principali(parametri, prelievo_annuo_da_usare):
    """
    Funzione core che esegue N simulazioni e restituisce i risultati grezzi.
    """
    n_simulazioni = parametri['n_simulazioni']
    mesi_totali = parametri['anni_totali'] * 12

    # Inizializzazione degli array aggregati
    patrimoni_tutte_le_run = np.zeros((n_simulazioni, mesi_totali + 1))
    patrimoni_reali_tutte_le_run = np.zeros((n_simulazioni, mesi_totali + 1))
    reddito_reale_annuo_tutte_le_run = np.zeros((n_simulazioni, parametri['anni_totali']))
    
    # Lista per contenere i dati annuali di OGNI simulazione
    tutti_i_dati_annuali_run = []
    
    # Contatori per statistiche
    totale_contributi_versati_nominale = np.zeros(n_simulazioni)
    guadagni_accumulo_nominale = np.zeros(n_simulazioni)
    drawdowns = np.zeros(n_simulazioni)
    sharpe_ratios = np.zeros(n_simulazioni)
    fallimenti = 0

    for i in range(n_simulazioni):
        risultati_run = _esegui_una_simulazione(parametri, prelievo_annuo_da_usare)
        
        patrimoni_tutte_le_run[i, :] = risultati_run['patrimoni_run']
        patrimoni_reali_tutte_le_run[i, :] = risultati_run['patrimoni_reali_run']
        reddito_reale_annuo_tutte_le_run[i, :] = risultati_run['reddito_annuo_reale']
        totale_contributi_versati_nominale[i] = risultati_run['totale_contributi_versati_nominale']
        guadagni_accumulo_nominale[i] = risultati_run['guadagni_accumulo']
        drawdowns[i] = risultati_run['drawdown']
        sharpe_ratios[i] = risultati_run['sharpe_ratio']
        if risultati_run['fallimento']:
            fallimenti += 1

        # Aggiungi i dati annuali di questa run alla lista
        tutti_i_dati_annuali_run.append(risultati_run['dati_annuali'])
            
    return (
        patrimoni_tutte_le_run,
        patrimoni_reali_tutte_le_run,
        reddito_reale_annuo_tutte_le_run,
        tutti_i_dati_annuali_run,
        {
            'totale_contributi_versati_nominale': totale_contributi_versati_nominale,
            'guadagni_accumulo_nominale': guadagni_accumulo_nominale,
            'drawdowns': drawdowns,
            'sharpe_ratios': sharpe_ratios,
            'fallimenti': fallimenti
        }
    )

def run_full_simulation(parametri):
    """
    Esegue la simulazione Monte Carlo completa, orchestrando migliaia di singole run.

    Questa funzione è il cuore del simulatore. Itera per il numero di simulazioni specificato,
    invocando `run_single_simulation` per ciascuna. Aggrega i risultati di tutte le run per
    calcolare statistiche robuste come la probabilità di fallimento, i percentili del patrimonio
    finale (nominale e reale) e il valore mediano dei guadagni ottenuti durante la fase di accumulo.

    Inoltre, identifica lo "scenario mediano" (la simulazione il cui risultato finale è più
    vicino alla mediana di tutte le simulazioni) e ne estrae i dati dettagliati, che verranno
    poi utilizzati per generare i grafici di dettaglio nell'interfaccia utente.

    Args:
        parametri (dict): Il dizionario contenente tutti i parametri di input.

    Returns:
        dict: Un dizionario contenente le statistiche aggregate, i dati per i grafici principali
              (l'evoluzione del patrimonio attraverso i percentili) e i dati annuali dettagliati
              dello scenario mediano.
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
    indici_fallimenti = []
    
    guadagni_accumulo_agg = np.zeros(n_sim)

    # Aggrega TUTTI i dati annuali per poter analizzare lo scenario mediano in dettaglio
    tutti_i_dati_annuali = {
        'prelievi_target_nominali': np.zeros((n_sim, num_anni + 1)),
        'prelievi_effettivi_nominali': np.zeros((n_sim, num_anni + 1)),
        'prelievi_effettivi_reali': np.zeros((n_sim, num_anni + 1)),
        'prelievi_da_banca_nominali': np.zeros((n_sim, num_anni + 1)),
        'prelievi_da_etf_nominali': np.zeros((n_sim, num_anni + 1)),
        'vendite_rebalance_nominali': np.zeros((n_sim, num_anni + 1)),
        'fp_liquidato_nominale': np.zeros((n_sim, num_anni + 1)),
        'pensioni_pubbliche_nominali': np.zeros((n_sim, num_anni + 1)),
        'pensioni_pubbliche_reali': np.zeros((n_sim, num_anni + 1)),
        'rendite_fp_nominali': np.zeros((n_sim, num_anni + 1)),
        'rendite_fp_reali': np.zeros((n_sim, num_anni + 1)),
        'saldo_banca_nominale': np.zeros((n_sim, num_anni + 1)),
        'saldo_etf_nominale': np.zeros((n_sim, num_anni + 1)),
        'saldo_banca_reale': np.zeros((n_sim, num_anni + 1)),
        'saldo_etf_reale': np.zeros((n_sim, num_anni + 1)),
        'saldo_fp_nominale': np.zeros((n_sim, num_anni + 1)),
        'saldo_fp_reale': np.zeros((n_sim, num_anni + 1)),
        'reddito_totale_reale': np.zeros((n_sim, num_anni + 1)),
        'variazione_patrimonio_percentuale': np.zeros((n_sim, num_anni + 1)),
        'rendimento_investimento_percentuale': np.zeros((n_sim, num_anni + 1))
    }
    contributi_totali_agg = np.zeros(n_sim)

    prelievo_annuo_da_usare = parametri['prelievo_annuo']
    calcolo_sostenibile_attivo = (
        parametri['strategia_prelievo'] == 'FISSO' and 
        parametri['prelievo_annuo'] == 0
    )
    prelievo_sostenibile_calcolato = None
    if calcolo_sostenibile_attivo:
        prelievo_sostenibile_calcolato = _calcola_prelievo_sostenibile(parametri)
        prelievo_annuo_da_usare = prelievo_sostenibile_calcolato

    # --- ESECUZIONE SIMULAZIONI PRINCIPALI ---
    # Eseguiamo SEMPRE la simulazione completa qui con il numero di simulazioni corretto
    # e il prelievo appena calcolato, per garantire che i risultati finali siano accurati.
    (
        patrimoni_tutte_le_run,
        patrimoni_reali_tutte_le_run,
        reddito_reale_annuo_tutte_le_run,
        tutti_i_dati_annuali_run,
        contatori_statistiche
    ) = _esegui_simulazioni_principali(parametri, prelievo_annuo_da_usare)

    # --- CALCOLO STATISTICHE FINALI ---
    patrimoni_finali_reali = patrimoni_reali_tutte_le_run[:, -1]
    patrimoni_finali_nominali = patrimoni_tutte_le_run[:, -1]
    
    # --- NUOVA LOGICA PER TROVARE LA SIMULAZIONE MEDIANA ---
    # 1. Calcola il valore mediano dei patrimoni finali reali
    valore_mediano = np.median(patrimoni_finali_reali)
    # 2. Trova l'indice della simulazione il cui patrimonio finale è più vicino alla mediana
    indice_mediano = np.abs(patrimoni_finali_reali - valore_mediano).argmin()
    # 3. Estrai i dati dettagliati di quella specifica simulazione
    dati_mediana_dettagliati = tutti_i_dati_annuali_run[indice_mediano]

    # Calcolo del patrimonio all'inizio dei prelievi
    idx_inizio_prelievo_mesi = parametri['anni_inizio_prelievo'] * 12
    patrimoni_inizio_prelievi_reali = patrimoni_reali_tutte_le_run[:, idx_inizio_prelievo_mesi]
    patrimoni_inizio_prelievi_nominali = patrimoni_tutte_le_run[:, idx_inizio_prelievo_mesi]
    
    # Filtra i drawdown e sharpe ratio degli scenari di successo per non distorcere le medie
    scenari_successo_mask = patrimoni_finali_reali > 1.0 # Considera successo se resta almeno 1€
    sharpe_ratios_successo = contatori_statistiche['sharpe_ratios'][scenari_successo_mask]

    statistiche = {
        'patrimonio_finale_mediano_nominale': np.median(patrimoni_finali_nominali),
        'patrimonio_finale_top_10_nominale': np.percentile(patrimoni_finali_nominali, 90),
        'patrimonio_finale_peggior_10_nominale': np.percentile(patrimoni_finali_nominali, 10),
        
        'patrimonio_finale_mediano_reale': np.median(patrimoni_finali_reali),
        'patrimonio_finale_top_10_reale': np.percentile(patrimoni_finali_reali, 90),
        'patrimonio_finale_peggior_10_reale': np.percentile(patrimoni_finali_reali, 10),

        'patrimonio_inizio_prelievi_mediano_nominale': np.median(patrimoni_inizio_prelievi_nominali),
        'patrimonio_inizio_prelievi_mediano_reale': np.median(patrimoni_inizio_prelievi_reali),

        'probabilita_fallimento': contatori_statistiche['fallimenti'] / n_sim,
        'drawdown_massimo_peggiore': np.min(contatori_statistiche['drawdowns']) if contatori_statistiche['drawdowns'].size > 0 else 0,
        'sharpe_ratio_medio': np.mean(sharpe_ratios_successo[np.isfinite(sharpe_ratios_successo)]) if sharpe_ratios_successo.size > 0 else 0,
        
        'patrimoni_reali_finali': patrimoni_finali_reali,
        
        'contributi_totali_versati_mediano_nominale': np.median(contatori_statistiche['totale_contributi_versati_nominale']),
        'guadagni_accumulo_mediano_nominale': np.median(contatori_statistiche['guadagni_accumulo_nominale']),
        
        'prelievo_sostenibile_calcolato': prelievo_sostenibile_calcolato
    }

    # --- CALCOLO STATISTICHE PRELIEVI ---
    # Calcola il reddito reale annuo medio solo sugli anni di prelievo e sugli scenari di successo
    anni_di_prelievo = parametri['anni_totali'] - parametri['anni_inizio_prelievo']
    if anni_di_prelievo > 0:
        idx_inizio_prelievo_anni = parametri['anni_inizio_prelievo']
        
        # Filtra gli scenari di successo (patrimonio finale > 0)
        scenari_successo_mask = patrimoni_finali_reali > 0
        redditi_scenari_successo = reddito_reale_annuo_tutte_le_run[scenari_successo_mask]
        
        # Considera solo gli anni di prelievo
        if redditi_scenari_successo.size > 0:
            redditi_fase_prelievo = redditi_scenari_successo[:, idx_inizio_prelievo_anni:]
            
            # Calcola la media solo dove il reddito è positivo, per ogni anno
            medie_annue = np.true_divide(redditi_fase_prelievo.sum(axis=0), (redditi_fase_prelievo > 0).sum(axis=0))
            medie_annue[np.isnan(medie_annue)] = 0 # Gestisci divisione per zero
            
            # La statistica finale è la media delle medie annuali
            totale_reale_medio_annuo = np.mean(medie_annue[medie_annue > 0]) if np.any(medie_annue > 0) else 0
        else:
            totale_reale_medio_annuo = 0
    else:
        totale_reale_medio_annuo = 0
        
    statistiche_prelievi = {
        'totale_reale_medio_annuo': totale_reale_medio_annuo
    }
    
    # --- PREPARAZIONE DATI PER I GRAFICI ---
    dati_grafici_principali = {
        "nominale": patrimoni_tutte_le_run,
        "reale": patrimoni_reali_tutte_le_run,
        "reddito_reale_annuo": reddito_reale_annuo_tutte_le_run
    }
    
    dati_grafici_avanzati = {
        "dati_mediana": dati_mediana_dettagliati
    }

    return {
        "statistiche": statistiche,
        "statistiche_prelievi": statistiche_prelievi,
        "dati_grafici_principali": dati_grafici_principali,
        "dati_grafici_avanzati": dati_grafici_avanzati
    } 

def _genera_evento_mercato_estremo(eventi_mercato_estremi):
    """
    Genera un evento di mercato estremo basato sulla configurazione.
    Restituisce un moltiplicatore del rendimento (es. 0.8 per -20%, 1.3 per +30%).
    """
    if eventi_mercato_estremi == 'DISABILITATI':
        return 1.0
    
    # Probabilità di evento estremo per mese
    prob_mensili = {
        'REALISTICI': 0.0017,      # ~2% annuo
        'FREQUENTI': 0.0042,       # ~5% annuo  
        'MOLTO_FREQUENTI': 0.0083  # ~10% annuo
    }
    
    prob = prob_mensili.get(eventi_mercato_estremi, 0.0)
    
    if np.random.random() < prob:
        # Genera un evento estremo
        tipo_evento = np.random.choice(['CRASH', 'BOOM'], p=[0.7, 0.3])  # 70% crash, 30% boom
        
        if tipo_evento == 'CRASH':
            # Crollo tra -15% e -35%
            crash_magnitude = np.random.uniform(0.15, 0.35)
            return 1.0 - crash_magnitude
        else:
            # Boom tra +20% e +50%
            boom_magnitude = np.random.uniform(0.20, 0.50)
            return 1.0 + boom_magnitude
    
    return 1.0 