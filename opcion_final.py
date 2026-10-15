#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 09:29:15 2026

@author: franciscotafur
"""

# -*- coding: utf-8 -*-
import requests
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import time
import warnings

# Ignorar advertencias de pandas/yfinance para limpiar la consola
warnings.filterwarnings("ignore")

# ==========================================
# 1. FUNCIONES MATEMÁTICAS Y DE FECHAS
# ==========================================

def build_term_structure(days, ivs):
    """Crea una curva de volatilidad interpolada (MIMICKING PRINCIPAL)."""
    days = np.array(days)
    ivs = np.array(ivs)
    sort_idx = days.argsort()
    days, ivs = days[sort_idx], ivs[sort_idx]
    spline = interp1d(days, ivs, kind='linear', fill_value="extrapolate")
    return lambda dte: ivs[0] if dte < days[0] else (ivs[-1] if dte > days[-1] else float(spline(dte)))

def get_next_friday():
    """Calcula la fecha del próximo viernes."""
    today = datetime.now().date()
    days_ahead = 4 - today.weekday()
    if days_ahead < 0:
        days_ahead += 7
    return (today + timedelta(days_ahead)).strftime("%Y-%m-%d")

# ==========================================
# 2. LÓGICA DE HISTORIAL DE EARNINGS (RV 4)
# ==========================================

def get_rv_4_earnings(ticker_obj):
    try:
        df_earnings = ticker_obj.earnings_dates
        if df_earnings is None or df_earnings.empty: return 0, "N/A"
        ahora = datetime.now(df_earnings.index.tz)
        
        proximo = df_earnings[df_earnings.index >= ahora].sort_index().head(1)
        horario = "N/A"
        if not proximo.empty:
            horario = "AMC" if proximo.index[0].hour >= 16 else "BMO"

        reportes_pasados = df_earnings[df_earnings.index < ahora].head(4)
        if reportes_pasados.empty: return 0, horario
        
        start_date = (reportes_pasados.index.min() - timedelta(days=15)).strftime('%Y-%m-%d')
        hist = ticker_obj.history(start=start_date)
        if hist.empty: return 0, horario
        
        gaps = []
        for fecha_reporte_tz in reportes_pasados.index:
            try:
                f_reporte = fecha_reporte_tz.date()
                fechas_disponibles = hist.index.date
                idx_actual = list(fechas_disponibles).index(f_reporte) if f_reporte in fechas_disponibles else [i for i, d in enumerate(fechas_disponibles) if d <= f_reporte][-1]
                
                if fecha_reporte_tz.hour >= 16: # AMC
                    if idx_actual + 1 >= len(hist): continue
                    p_prev, p_post = hist['Close'].iloc[idx_actual], hist['Open'].iloc[idx_actual + 1]
                else: # BMO
                    if idx_actual - 1 < 0: continue
                    p_prev, p_post = hist['Close'].iloc[idx_actual - 1], hist['Open'].iloc[idx_actual]
                gaps.append(abs((p_post - p_prev) / p_prev))
            except: continue
        return (np.mean(gaps) * 100) if gaps else 0, horario
    except: return 0, "N/A"

# ==========================================
# 3. EXTRACCIÓN Y CÁLCULO POR SÍMBOLO
# ==========================================

def analyze_symbol(symbol):
    try:
        stock = yf.Ticker(symbol)
        next_friday = get_next_friday()
        if next_friday not in stock.options: return None

        hist = stock.history(period='3mo')
        if hist.empty: return None
        price = hist['Close'].iloc[-1]
        
        chain = stock.option_chain(next_friday)
        c_atm = chain.calls.loc[(chain.calls['strike'] - price).abs().idxmin()]
        p_atm = chain.puts.loc[(chain.puts['strike'] - price).abs().idxmin()]

        def calc_spread(opt):
            mid = (opt['bid'] + opt['ask']) / 2
            return ((opt['ask'] - opt['bid']) / mid * 100) if mid > 0 else 0

        rv_4, horario = get_rv_4_earnings(stock)
        em_pct = ((c_atm['lastPrice'] + p_atm['lastPrice']) / price) * 100
        
        # --- LÓGICA DE SLOPE E IV NETO (MIMICKING SCANNER PRINCIPAL) ---
        ts_slope, iv_neto = 0, 0
        if len(stock.options) >= 2:
            try:
                dtes, ivs = [], []
                # Analizar los primeros 3 vencimientos
                for exp in list(stock.options)[:3]:
                    ch = stock.option_chain(exp)
                    c_iv = ch.calls.iloc[(ch.calls['strike'] - price).abs().idxmin()]['impliedVolatility']
                    p_iv = ch.puts.iloc[(ch.puts['strike'] - price).abs().idxmin()]['impliedVolatility']
                    
                    dte = (datetime.strptime(exp, "%Y-%m-%d").date() - datetime.now().date()).days
                    dtes.append(max(dte, 0.5))
                    ivs.append((c_iv + p_iv) / 2.0)
                
                if len(dtes) >= 2:
                    term_spline = build_term_structure(dtes, ivs)
                    iv_vendida = ivs[0]
                    iv_comprada = term_spline(45)
                    
                    # Fórmulas idénticas a Scanner Principal
                    ts_slope = ((iv_comprada - iv_vendida) / (45 - dtes[0])) * 100
                    iv_neto = (iv_vendida - iv_comprada) * 100
            except: pass

        return {
            "Ticker": symbol,
            "Precio": round(price, 2),
            "Horario": horario,
            "EM Neto (%)": round(em_pct - rv_4, 2),
            "Expected Move (%)": round(em_pct, 2),
            "RV 4 Hist (%)": round(rv_4, 2),
            "IV Call ATM (%)": round(c_atm['impliedVolatility'] * 100, 2),
            "IV Put ATM (%)": round(p_atm['impliedVolatility'] * 100, 2),
            "IV Neto (%)": round(iv_neto, 2),
            "Slope (%)": round(ts_slope, 4),
            "Spread Call (%)": round(calc_spread(c_atm), 2),
            "Spread Put (%)": round(calc_spread(p_atm), 2),
            "OI Call ATM": int(c_atm['openInterest']),
            "OI Put ATM": int(p_atm['openInterest']),
            "Vol Total ATM": int((c_atm['volume'] or 0) + (p_atm['volume'] or 0))
        }
    except: return None

# ==========================================
# 4. COORDINADOR DEL ESCÁNER Y EXPORTACIÓN
# ==========================================

def run_scanner(fecha_dmy):
    f_iso = datetime.strptime(fecha_dmy, '%d/%m/%Y').strftime('%Y-%m-%d')
    url = f"https://api.nasdaq.com/api/calendar/earnings?date={f_iso}"
    headers = {"User-Agent": "Mozilla/5.0", "Origin": "https://www.nasdaq.com", "Referer": "https://www.nasdaq.com/"}
    
    try:
        r = requests.get(url, headers=headers).json()
        symbols_data = {row['symbol']: row.get('time', 'N/A') for row in r['data']['rows'] if row.get('symbol')}
    except: return print("Error conectando con Nasdaq API.")

    results = []
    total = len(symbols_data)
    print(f"--- Escaneando {total} símbolos para {fecha_dmy} (Logic: Mimicking Principal) ---")
    
    for i, (s, t) in enumerate(symbols_data.items(), 1):
        print(f"[{i}/{total}] {s}           ", end="\r")
        res = analyze_symbol(s)
        if res:
            if res["Horario"] == "N/A":
                res["Horario"] = "AMC" if "after" in t.lower() else ("BMO" if "before" in t.lower() else "N/A")
            results.append(res)
        if i % 5 == 0: time.sleep(0.1)
    
    if results:
        df = pd.DataFrame(results).sort_values(by="EM Neto (%)", ascending=False)
        filename = f"Scanner_Selling_Earnings_Fixed_{f_iso}.xlsx"
        df.to_excel(filename, index=False)
        print(f"\n\n✅ ¡Proceso finalizado! Archivo: {filename}")
        print(df[["Ticker", "EM Neto (%)", "IV Neto (%)", "Slope (%)", "Horario"]].head(10).to_string(index=False))
    else:
        print("\n\n❌ No se encontraron acciones con opciones para este viernes.")

if __name__ == "__main__":
    # Fecha de análisis (DD/MM/YYYY)
    FECHA_ANALISIS = "14/10/2026"
    run_scanner(FECHA_ANALISIS)