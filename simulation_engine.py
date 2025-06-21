import numpy as np
import pandas as pd

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
    if parametri['costo_fisso_deposito_titoli'] < 0:
        raise ValueError("Il costo fisso deposito titoli non può essere negativo")
    if parametri['usa_fp']:
        if not (0 <= parametri['fp_rendimento_netto'] <= 1):
            raise ValueError("Rendimento netto FP deve essere tra 0 e 1")
        if parametri['fp_capitale_iniziale'] < 0:
            raise ValueError("Capitale iniziale FP non può essere negativo")
        if parametri['fp_contributo_mensile'] < 0:
            raise ValueError("Contributo mensile FP non può essere negativo")
        if not (0 <= parametri['fp_aliquota_tassazione_finale'] <= 1):
            raise ValueError("Aliquota tassazione finale FP deve essere tra 0 e 1")

def _esegui_una_simulazione(params):
    """
    Esegue una singola traiettoria della simulazione Monte Carlo.
    Restituisce un dizionario contenente i dataframe dei risultati per questa singola esecuzione.
    """
    # Setup iniziale
    eta_iniziale = params['eta_iniziale']
    mesi_totali = params['anni_totali'] * 12
    eta_pensionamento = eta_iniziale + params['anni_inizio_prelievo']

    # Calcolo parametri aggregati del portafoglio
    etf_portfolio = params['etf_portfolio']
    weights = etf_portfolio['Allocazione (%)'] / 100
    rendimento_medio_portfolio = np.sum(weights * etf_portfolio['Rendimento Atteso (%)']) / 100
    volatilita_portfolio = np.sum(weights * etf_portfolio['Volatilità Attesa (%)']) / 100
    ter_portfolio = np.sum(weights * etf_portfolio['TER (%)']) / 100

    # Stato della simulazione
    stato = {
        'conto_corrente': params['capitale_iniziale'],
        'patrimonio_etf': params['etf_iniziale'],
        'cost_basis_etf': params['etf_iniziale'],
        'fp_valore': params['fp_capitale_iniziale'],
        'fp_rendita_annua_reale': 0,
        'fp_liquidato': False,
        'indice_prezzi': 1.0,
        'prelievo_annuo_nominale_corrente': 0,
    }

    # Dataframe per tracciare i dati anno per anno
    colonne_annuali = [
        'anno', 'eta', 'prelievo_nominale_obiettivo', 'prelievo_nominale_effettivo', 'prelievo_da_liquidita', 
        'prelievo_da_vendita_etf', 'vendita_per_rebalance', 'liquidazione_capitale_fp', 
        'prelievo_reale_effettivo', 'pensione_pubblica_reale', 'rendita_fp_reale', 'totale_entrate_reali', 
        'saldo_conto_fine_anno_reale', 'valore_etf_fine_anno_reale', 'patrimonio_totale_reale'
    ]
    dati_annuali_list = []
    
    patrimonio_mensile_reale = np.zeros(mesi_totali + 1)
    patrimonio_mensile_nominale = np.zeros(mesi_totali + 1)
    patrimonio_mensile_reale[0] = (stato['conto_corrente'] + stato['patrimonio_etf']) / stato['indice_prezzi']
    patrimonio_mensile_nominale[0] = stato['conto_corrente'] + stato['patrimonio_etf']

    for mese in range(1, mesi_totali + 1):
        anno = (mese - 1) // 12
        eta_attuale = eta_iniziale + anno
        is_fine_anno = (mese % 12 == 0)

        # --- OPERAZIONI MENSILI ---
        
        # 1. Contribuzioni (fase di accumulo)
        if eta_attuale < eta_pensionamento:
            stato['conto_corrente'] += params['contributo_mensile_banca']
            stato['patrimonio_etf'] += params['contributo_mensile_etf']
            stato['cost_basis_etf'] += params['contributo_mensile_etf']

        # 2. Fondo Pensione (accumulo)
        if params['usa_fp'] and not stato['fp_liquidato']:
            stato['fp_valore'] += params['fp_contributo_mensile']
        
        # 3. Entrate passive
        entrate_passive_mese = 0
        if params['usa_pensione_pubblica'] and eta_attuale >= params['eta_inizio_pensione_pubblica']:
            entrata_reale = params['pensione_pubblica_importo_annuo'] / 12
            entrate_passive_mese += entrata_reale * stato['indice_prezzi']
        if stato['fp_rendita_annua_reale'] > 0:
            entrata_reale = stato['fp_rendita_annua_reale'] / 12
            entrate_passive_mese += entrata_reale * stato['indice_prezzi']
        stato['conto_corrente'] += entrate_passive_mese

        # 4. Calcolo prelievo annuale (se inizio anno fiscale in pensione)
        if eta_attuale >= eta_pensionamento and (mese - 1 - params['anni_inizio_prelievo'] * 12) % 12 == 0:
            patrimonio_per_prelievo = stato['conto_corrente'] + stato['patrimonio_etf']
            
            if params['strategia_prelievo'] == 'FISSO':
                stato['prelievo_annuo_nominale_corrente'] = params['prelievo_annuo'] * stato['indice_prezzi']
            
            else: # REGOLA_4_PERCENTO / GUARDRAIL
                if stato['prelievo_annuo_nominale_corrente'] == 0: # Primo prelievo
                     stato['prelievo_annuo_nominale_corrente'] = patrimonio_per_prelievo * params['percentuale_prelievo']
                else: # Anni successivi
                    if params['strategia_prelievo'] == 'REGOLA_4_PERCENTO':
                        stato['prelievo_annuo_nominale_corrente'] = patrimonio_per_prelievo * params['percentuale_prelievo']
                    elif params['strategia_prelievo'] == 'GUARDRAIL':
                        prelievo_aggiornato_inflazione = stato['prelievo_annuo_nominale_corrente'] * (1 + params['inflazione'])
                        tasso_attuale = prelievo_aggiornato_inflazione / patrimonio_per_prelievo if patrimonio_per_prelievo > 0 else 0
                        soglia_sup = params['percentuale_prelievo'] * (1 + params['guardrail_superiore'])
                        soglia_inf = params['percentuale_prelievo'] * (1 - params['guardrail_inferiore'])
                        
                        if tasso_attuale > soglia_sup:
                            stato['prelievo_annuo_nominale_corrente'] = prelievo_aggiornato_inflazione * 0.9
                        elif tasso_attuale < soglia_inf:
                            stato['prelievo_annuo_nominale_corrente'] = prelievo_aggiornato_inflazione * 1.1
                        else:
                             stato['prelievo_annuo_nominale_corrente'] = prelievo_aggiornato_inflazione
        
        # 5. Esecuzione prelievo mensile
        prelievo_mensile = stato['prelievo_annuo_nominale_corrente'] / 12 if eta_attuale >= eta_pensionamento else 0
        prelievo_da_cc = min(stato['conto_corrente'], prelievo_mensile)
        stato['conto_corrente'] -= prelievo_da_cc
        fabbisogno_da_etf = prelievo_mensile - prelievo_da_cc
        
        vendita_netta_etf = 0
        if fabbisogno_da_etf > 0 and stato['patrimonio_etf'] > 0:
            # Calcolo tassazione sul capital gain
            cost_basis_ratio = stato['cost_basis_etf'] / stato['patrimonio_etf'] if stato['patrimonio_etf'] > 0 else 1
            plusvalenza_ratio = 1 - cost_basis_ratio
            
            vendita_lorda = fabbisogno_da_etf / (1 - plusvalenza_ratio * params['tassazione_capital_gain']) if (1 - plusvalenza_ratio * params['tassazione_capital_gain']) > 0 else float('inf')
            vendita_lorda = min(vendita_lorda, stato['patrimonio_etf'])
            
            plusvalenza = vendita_lorda * plusvalenza_ratio
            tasse = plusvalenza * params['tassazione_capital_gain']
            vendita_netta_etf = vendita_lorda - tasse
            stato['conto_corrente'] += vendita_netta_etf
            
            # Aggiorna valore e cost basis
            stato['patrimonio_etf'] -= vendita_lorda
            stato['cost_basis_etf'] -= (vendita_lorda * cost_basis_ratio)
        
        prelievo_effettivo_mese = prelievo_da_cc + vendita_netta_etf

        # 6. Applica rendimenti di mercato e inflazione
        # Ripristino la formula corretta per la Geometric Brownian Motion
        rendimento_mese = np.random.lognormal(
            rendimento_medio_portfolio / 12 - 0.5 * (volatilita_portfolio / np.sqrt(12))**2, 
            volatilita_portfolio / np.sqrt(12)
        )
        stato['patrimonio_etf'] *= rendimento_mese

        if params['usa_fp'] and not stato['fp_liquidato']:
             # Assumiamo una volatilità fissa del 5% per il fondo pensione come prima
             volatilita_fp = 0.05 
             rendimento_fp_mese = np.random.lognormal(
                 params['fp_rendimento_netto'] / 12 - 0.5 * (volatilita_fp / np.sqrt(12))**2, 
                 volatilita_fp / np.sqrt(12)
             )
             stato['fp_valore'] *= rendimento_fp_mese
        
        stato['indice_prezzi'] *= (1 + np.random.normal(params['inflazione'] / 12, 0.005))
        
        patrimonio_mensile_reale[mese] = (stato['conto_corrente'] + stato['patrimonio_etf']) / stato['indice_prezzi']
        patrimonio_mensile_nominale[mese] = stato['conto_corrente'] + stato['patrimonio_etf']

        # --- OPERAZIONI DI FINE ANNO ---
        if is_fine_anno:
            # Costi e Tasse Annuali
            stato['patrimonio_etf'] -= stato['patrimonio_etf'] * ter_portfolio
            if stato['conto_corrente'] > 5000:
                stato['conto_corrente'] -= params['imposta_bollo_liquidita']
            if stato['patrimonio_etf'] > 0:
                stato['patrimonio_etf'] -= stato['patrimonio_etf'] * params['imposta_bollo_titoli']

            stato['conto_corrente'] -= params['costo_fisso_deposito_titoli']

            # Liquidazione Fondo Pensione
            liquidazione_fp_anno = 0
            if params['usa_fp'] and eta_attuale >= params['fp_eta_liquidazione'] and not stato['fp_liquidato']:
                capitale_da_liquidare = stato['fp_valore'] * (params['fp_perc_liquidazione_capitale'] / 100.0)
                tasse_fp = capitale_da_liquidare * params['fp_aliquota_tassazione_finale']
                liquidazione_fp_anno = capitale_da_liquidare - tasse_fp
                stato['conto_corrente'] += liquidazione_fp_anno
                
                montante_per_rendita = stato['fp_valore'] - capitale_da_liquidare
                # Semplificazione: coefficiente di conversione basato su aspettativa di vita residua
                anni_residui_attesi = max(1, 95 - eta_attuale)
                stato['fp_rendita_annua_reale'] = montante_per_rendita / anni_residui_attesi / stato['indice_prezzi']
                stato['fp_liquidato'] = True

            # Rebalancing (Glidepath)
            vendita_rebalance = 0
            if params['use_glidepath'] and params['start_glidepath_eta'] <= eta_attuale < params['end_glidepath_eta']:
                # Calcola l'allocazione target di ETF per l'età attuale
                progresso_glidepath = (eta_attuale - params['start_glidepath_eta']) / (params['end_glidepath_eta'] - params['start_glidepath_eta'])
                allocazione_etf_target = 1 - (1 - (params['final_equity_percentage'] / 100.0)) * progresso_glidepath
                
                patrimonio_totale_investibile = stato['conto_corrente'] + stato['patrimonio_etf']
                valore_etf_target = patrimonio_totale_investibile * allocazione_etf_target
                valore_etf_attuale = stato['patrimonio_etf']

                delta_etf = valore_etf_attuale - valore_etf_target
                if delta_etf > 0: # Bisogna vendere ETF per passare a liquidità
                    vendita_rebalance = delta_etf
                    # Semplificazione della tassazione anche qui
                    cost_basis_ratio = stato['cost_basis_etf'] / stato['patrimonio_etf'] if stato['patrimonio_etf'] > 0 else 1
                    plusvalenza_ratio = 1 - cost_basis_ratio
                    
                    tasse_rebalance = vendita_rebalance * plusvalenza_ratio * params['tassazione_capital_gain']
                    ricavo_netto_rebalance = vendita_rebalance - tasse_rebalance
                    
                    stato['conto_corrente'] += ricavo_netto_rebalance
                    stato['patrimonio_etf'] -= vendita_rebalance
                    stato['cost_basis_etf'] -= (vendita_rebalance * cost_basis_ratio)

            # Salva dati annuali per il dataframe di dettaglio
            dati_anno_corrente = {
                'anno': anno + 1,
                'eta': eta_attuale,
                'prelievo_nominale_obiettivo': stato['prelievo_annuo_nominale_corrente'],
                'prelievo_nominale_effettivo': prelievo_effettivo_mese * 12, # Stima annuale
                'prelievo_da_liquidita': prelievo_da_cc * 12, # Stima annuale
                'prelievo_da_vendita_etf': vendita_netta_etf * 12, # Stima annuale
                'vendita_per_rebalance': vendita_rebalance,
                'liquidazione_capitale_fp': liquidazione_fp_anno,
                'prelievo_reale_effettivo': (prelievo_effettivo_mese * 12) / stato['indice_prezzi'],
                'pensione_pubblica_reale': (params['pensione_pubblica_importo_annuo'] if params['usa_pensione_pubblica'] and eta_attuale >= params['eta_inizio_pensione_pubblica'] else 0),
                'rendita_fp_reale': stato['fp_rendita_annua_reale'] if stato['fp_rendita_annua_reale'] > 0 else 0,
                'totale_entrate_reali': ((prelievo_effettivo_mese * 12) / stato['indice_prezzi']) + \
                                      (params['pensione_pubblica_importo_annuo'] if params['usa_pensione_pubblica'] and eta_attuale >= params['eta_inizio_pensione_pubblica'] else 0) + \
                                      (stato['fp_rendita_annua_reale'] if stato['fp_rendita_annua_reale'] > 0 else 0),
                'saldo_conto_fine_anno_reale': stato['conto_corrente'] / stato['indice_prezzi'],
                'valore_etf_fine_anno_reale': stato['patrimonio_etf'] / stato['indice_prezzi'],
                'patrimonio_totale_reale': (stato['conto_corrente'] + stato['patrimonio_etf']) / stato['indice_prezzi']
            }
            dati_annuali_list.append(dati_anno_corrente)
    
    return patrimonio_mensile_reale, patrimonio_mensile_nominale, pd.DataFrame(dati_annuali_list, columns=colonne_annuali)


def run_full_simulation(params):
    """
    Esegue la simulazione Monte Carlo completa e calcola le statistiche.
    """
    tutti_i_risultati_mensili = []
    lista_df_dettaglio = []
    tutti_i_risultati_nominali = []

    for i in range(params['n_simulazioni']):
        patrimonio_reale, patrimonio_nominale, df_dettaglio = _esegui_una_simulazione(params)
        tutti_i_risultati_mensili.append(patrimonio_reale)
        tutti_i_risultati_nominali.append(patrimonio_nominale)
        lista_df_dettaglio.append(df_dettaglio)

    df_risultati_reali = pd.DataFrame(tutti_i_risultati_mensili).transpose()
    df_risultati_reali.columns = [f'Sim_{i+1}' for i in range(params['n_simulazioni'])]
    
    df_risultati_nominali = pd.DataFrame(tutti_i_risultati_nominali).transpose()
    df_risultati_nominali.columns = [f'Sim_{i+1}' for i in range(params['n_simulazioni'])]
    
    # Crea un DataFrame con i dati di dettaglio di tutte le simulazioni
    df_dettaglio_completo = pd.concat(lista_df_dettaglio, keys=range(params['n_simulazioni']), names=['Sim_Num', 'Anno_Index'])
    
    # Crea un DataFrame con i redditi di tutte le simulazioni partendo dai dati di dettaglio raccolti
    df_redditi_reali_annui = df_dettaglio_completo.reset_index().pivot(
        index='Anno_Index', columns='Sim_Num', values='totale_entrate_reali'
    )
    df_redditi_reali_annui.columns = [f'Sim_{i+1}' for i in range(params['n_simulazioni'])]

    # Calcolo statistiche principali
    patrimoni_iniziale = params['capitale_iniziale'] + params['etf_iniziale']
    patrimoni_reali_finali = df_risultati_reali.iloc[-1]
    patrimoni_nominali_finali = df_risultati_nominali.iloc[-1]
    
    # Calcolo drawdown massimo (Maximum Drawdown, MDD) in modo più robusto.
    # Calcoliamo il MDD per ogni singola simulazione, poi prendiamo la mediana di questi valori.
    # Questo dà una rappresentazione più fedele del drawdown che l'investitore "mediano" sperimenta.
    drawdowns_reali = []
    # Usiamo i dati annuali per un calcolo più stabile
    df_risultati_reali_annuali = df_risultati_reali.iloc[::12, :] 
    for col in df_risultati_reali_annuali.columns:
        serie = df_risultati_reali_annuali[col]
        peak = serie.expanding(min_periods=1).max()
        drawdown = (serie - peak) / peak
        drawdowns_reali.append(drawdown.min())
    
    # Usiamo la mediana dei drawdown per una metrica più stabile e rappresentativa
    drawdown_mediano = np.median(drawdowns_reali) if drawdowns_reali else 0
    
    # Calcolo Sharpe ratio medio (semplificato)
    rendimenti_annuali_sim = []
    for col in df_risultati_reali.columns:
        serie = df_risultati_reali[col]
        if len(serie) > 12:
            rendimenti = []
            for i in range(12, len(serie), 12):
                if i < len(serie):
                    rendimento = (serie.iloc[i] - serie.iloc[i-12]) / serie.iloc[i-12]
                    rendimenti.append(rendimento)
            if rendimenti:
                rendimenti_annuali_sim.extend(rendimenti)
    
    sharpe_ratio_medio = np.mean(rendimenti_annuali_sim) / np.std(rendimenti_annuali_sim) if rendimenti_annuali_sim and np.std(rendimenti_annuali_sim) > 0 else 0
    
    stats = {
        'patrimonio_iniziale': patrimoni_iniziale,
        'probabilita_fallimento': (patrimoni_reali_finali <= 1).mean(),
        'patrimonio_finale_mediano_reale': patrimoni_reali_finali.median(),
        'patrimonio_finale_mediano_nominale': patrimoni_nominali_finali.median(),
        'patrimonio_finale_top_10_nominale': patrimoni_nominali_finali.quantile(0.90),
        'patrimonio_finale_peggior_10_nominale': patrimoni_nominali_finali.quantile(0.10),
        'drawdown_massimo_mediano': abs(drawdown_mediano),
        'sharpe_ratio_medio': sharpe_ratio_medio,
        'patrimoni_reali_finali': patrimoni_reali_finali.to_dict(),
    }

    # Calcolo tenore di vita mediano e sue componenti da TUTTE le simulazioni per coerenza
    totale_reale_medio_annuo_mediano = 0
    prelievi_reali_mediani = 0
    pensioni_reali_mediane = 0
    rendite_fp_reali_mediane = 0

    if not df_redditi_reali_annui.empty:
        anni_pensione_start_index = params['anni_inizio_prelievo']
        if anni_pensione_start_index < len(df_redditi_reali_annui.index):
            # Isola i dati del reddito solo per gli anni di pensione
            redditi_in_pensione = df_redditi_reali_annui.iloc[anni_pensione_start_index:]
            
            # Isola i dati delle singole componenti del reddito per la pensione
            df_dettaglio_pensione = df_dettaglio_completo[df_dettaglio_completo['eta'] >= params['eta_iniziale'] + params['anni_inizio_prelievo']]

            if not redditi_in_pensione.empty:
                # Calcola la media annua per ogni simulazione
                medie_redditi_per_sim = redditi_in_pensione.mean(axis=0)
                totale_reale_medio_annuo_mediano = medie_redditi_per_sim.median()

            if not df_dettaglio_pensione.empty:
                # Calcola la media per ogni componente per ogni simulazione, poi la mediana di queste medie
                prelievi_reali_mediani = df_dettaglio_pensione.groupby('Sim_Num')['prelievo_reale_effettivo'].mean().median()
                pensioni_reali_mediane = df_dettaglio_pensione.groupby('Sim_Num')['pensione_pubblica_reale'].mean().median()
                rendite_fp_reali_mediane = df_dettaglio_pensione.groupby('Sim_Num')['rendita_fp_reale'].mean().median()

    # Sostituzione delle statistiche basate su una sola simulazione con quelle aggregate
    stats_prelievi = {
        'totale_reale_medio_annuo': totale_reale_medio_annuo_mediano,
        'prelievo_reale_medio': prelievi_reali_mediani,
        'pensione_pubblica_reale_annua': pensioni_reali_mediane,
        'rendita_fp_reale_media': rendite_fp_reali_mediane
    }
    
    # Preparazione dati per i grafici
    dati_grafici_principali = {
        'reale': df_risultati_reali.to_dict('split'),
        'nominale': df_risultati_nominali.to_dict('split'),
        'reddito_reale_annuo': df_redditi_reali_annui.to_dict('split')
    }
    
    # Prepara i dati di dettaglio per i grafici basati sulla simulazione mediana
    mediana_idx = patrimoni_reali_finali.sort_values().index[len(patrimoni_reali_finali) // 2]
    sim_num_mediana = int(mediana_idx.split('_')[1]) - 1
    df_prelievi = lista_df_dettaglio[sim_num_mediana] if sim_num_mediana < len(lista_df_dettaglio) else pd.DataFrame()

    dati_grafici_avanzati = {
        'dati_mediana': {
            'prelievi_target_nominali': df_prelievi['prelievo_nominale_obiettivo'].tolist() if not df_prelievi.empty else [],
            'prelievi_effettivi_nominali': df_prelievi['prelievo_nominale_effettivo'].tolist() if not df_prelievi.empty else [],
            'prelievi_da_banca_nominali': df_prelievi['prelievo_da_liquidita'].tolist() if not df_prelievi.empty else [],
            'prelievi_da_etf_nominali': df_prelievi['prelievo_da_vendita_etf'].tolist() if not df_prelievi.empty else [],
            'vendite_rebalance_nominali': df_prelievi['vendita_per_rebalance'].tolist() if not df_prelievi.empty else [],
            'fp_liquidato_nominale': df_prelievi['liquidazione_capitale_fp'].tolist() if not df_prelievi.empty else [],
            'prelievi_effettivi_reali': df_prelievi['prelievo_reale_effettivo'].tolist() if not df_prelievi.empty else [],
            'pensioni_pubbliche_reali': df_prelievi['pensione_pubblica_reale'].tolist() if not df_prelievi.empty else [],
            'rendite_fp_reali': df_prelievi['rendita_fp_reale'].tolist() if not df_prelievi.empty else [],
            'saldo_banca_reale': df_prelievi['saldo_conto_fine_anno_reale'].tolist() if not df_prelievi.empty else [],
            'saldo_etf_reale': df_prelievi['valore_etf_fine_anno_reale'].tolist() if not df_prelievi.empty else [],
            'saldo_fp_reale': [0] * len(df_prelievi) if not df_prelievi.empty else []  # Semplificazione
        }
    }
    
    # Struttura finale dei risultati
    risultati = {
        'statistiche': stats,
        'statistiche_prelievi': stats_prelievi,
        'dati_grafici_principali': dati_grafici_principali,
        'dati_grafici_avanzati': dati_grafici_avanzati
    }
    
    return risultati 