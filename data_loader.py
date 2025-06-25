"""
data_loader.py

Modulo per il caricamento e la gestione di dati storici finanziari e demografici.
- S&P 500, altri indici azionari
- Inflazione storica (USA, Europa, Italia)
- Tabelle di longevità
- Eventuali dati di crisi/stress
"""

import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self):
        """Inizializza il data loader e prepara i dataset."""
        self.market_data = None
        self.inflation_data = None
        self.longevity_data = None

    def load_market_data(self, source='sp500.csv'):
        """Carica dati storici di mercato da file o API."""
        # TODO: implementare caricamento reale
        self.market_data = pd.DataFrame()
        return self.market_data

    def load_inflation_data(self, source='inflation.csv'):
        """Carica dati storici di inflazione da file o API."""
        # TODO: implementare caricamento reale
        self.inflation_data = pd.DataFrame()
        return self.inflation_data

    def load_longevity_data(self, source='longevity.csv'):
        """Carica tabelle di longevità da file o API."""
        # TODO: implementare caricamento reale
        self.longevity_data = pd.DataFrame()
        return self.longevity_data

    def get_market_returns(self):
        """Restituisce una serie di rendimenti annuali dal dataset di mercato."""
        # TODO: calcolo reale
        return np.array([])

    def get_inflation_series(self):
        """Restituisce una serie di inflazione annuale dal dataset."""
        # TODO: calcolo reale
        return np.array([])

    def get_longevity_distribution(self):
        """Restituisce una distribuzione di longevità (es. probabilità di sopravvivenza per età)."""
        # TODO: calcolo reale
        return np.array([])

    def load_sample_historical_data(self, years=60):
        """Restituisce dati storici simulati di rendimenti azionari e inflazione per demo/test."""
        np.random.seed(42)
        returns = np.random.normal(loc=0.07, scale=0.15, size=years)  # 7% medio, 15% vol
        inflation = np.random.normal(loc=0.025, scale=0.01, size=years)  # 2.5% medio, 1% vol
        df = pd.DataFrame({
            'year': np.arange(1960, 1960+years),
            'equity_return': returns,
            'inflation': inflation
        })
        return df 