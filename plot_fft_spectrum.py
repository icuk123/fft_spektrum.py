#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Plot FFT Spectrum dari VIBRO.CSV
# Setiap baris CSV = satu spike vertikal di posisi frekuensinya
# Layout 2x2: [Overall RMS Trend] [Spektrum X] [Spektrum Y] [Spektrum Z]

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
NOMINAL_RPM = 1800
CSV_PATH    = r"D:\SEMESTER7\KERJAPRAKTIK\VIBRO.CSV"
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def iso_zone(v):
    if v < 1.4:
        return "ZONA A (Good)", "#16A34A"
    elif v < 2.8:
        return "ZONA B (Normal)", "#D97706"
    elif v < 4.5:
        return "ZONA C (Alert)", "#EA580C"
    else:
        return "ZONA D (DANGER)", "#DC2626"


def velocity_rms(f_hz, m_mg):
    if f_hz <= 0 or m_mg <= 0:
        return 0.0
    a_ms2 = (m_mg / 1000.0) * 9.81
    v_ms = a_ms2 / (2.0 * np.pi * f_hz)
    return round(v_ms * 1000.0 / np.sqrt(2.0), 3)  # mm/s RMS


def get_all_rows(f_col, m_col):
    """
    Kembalikan SEMUA pasangan (freq, mag) dari setiap baris CSV yang valid.
    Setiap baris = satu titik data individu.
    """
    valid = (f_col > 0) & (m_col > 0)
    freqs = f_col[valid].values.astype(float)
    mags = m_col[valid].values.astype(float)
    return freqs, mags


def get_top_peaks(freqs, mags, top=5, min_gap_hz=5.0):
    """
    Dari semua data mentah, pilih puncak tertinggi untuk dilabeli.
    Pastikan jarak antar label minimal min_gap_hz Hz agar tidak bertumpuk.
    """
    if len(freqs) == 0:
        return []

    # Kumpulkan max per frekuensi (dibulatkan 1 Hz)
    peak_dict = {}
    for f, m in zip(freqs, mags):
        key = round(f, 1)
        if key not in peak_dict or m > peak_dict[key]:
            peak_dict[key] = m

    sorted_peaks = sorted(peak_dict.items(), key=lambda x: x[1], reverse=True)

    selected = []
    for f, m in sorted_peaks:
        if m < 1.0:
            continue
        if all(abs(f - sf) >= min_gap_hz for sf, _ in selected):
            selected.append((f, m))
        if len(selected) >= top:
            break
    return selected


def label_peaks(ax, freqs, mags, f1x, color, y_max):
    peaks = get_top_peaks(freqs, mags)
    for f, m in peaks:
        ratio = f / f1x
        n = round(ratio)
        tag = f"{n}X" if n >= 1 and abs(ratio - n) < 0.15 else f"{ratio:.1f}X"
        ax.annotate(
            f"{tag}\n{m:.0f} mg",
            xy=(f, m),
            xytext=(f, m + y_max * 0.07),
            ha='center', va='bottom',
            fontsize=8.0, fontweight='bold', color=color,
            arrowprops=dict(arrowstyle='->', color=color, lw=1.0)
        )


def style_ax(ax):
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for s in ['left', 'bottom']:
        ax.spines[s].set_color('#0F172A')
        ax.spines[s].set_linewidth(1.8)
    ax.grid(axis='y', color='#E2E8F0', lw=0.6, ls='-')
    ax.set_axisbelow(True)
    ax.tick_params(colors='#0F172A', labelsize=8.5, length=4, width=1.2)


def plot_spectrum_ax(ax, freqs, mags, color, f_lim, y_max, title, ylabel, y_tick):
    if len(freqs) > 0:
        # lw=0.7 tipis agar spike padat tidak saling menutup
        # alpha=0.65 transparan agar overlap frekuensi tetap terbaca
        ax.vlines(freqs, 0, mags, colors=color, lw=0.7, alpha=0.65)
    ax.set_title(title, fontsize=9.5, fontweight='bold', color='#0F172A', pad=7)
    ax.set_xlabel("Frekuensi (Hz)", fontsize=9, fontweight='bold',
                 color='#0F172A', labelpad=3)
    ax.set_ylabel(ylabel, fontsize=8.5, fontweight='bold', color=color)
    ax.set_xlim(0, f_lim)
    ax.set_ylim(0, y_max)
    ax.xaxis.set_major_locator(MultipleLocator(
        100 if f_lim > 500 else (50 if f_lim > 200 else 10)
    ))
    ax.yaxis.set_major_locator(MultipleLocator(y_tick))


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def main():
    if not os.path.exists(CSV_PATH):
        print("[ERROR] File tidak ditemukan:", CSV_PATH)
        return

    df = pd.read_csv(CSV_PATH)
    n = len(df)
    print(f"[OK] {n:,} baris dibaca dari {CSV_PATH}")

    f1x = NOMINAL_RPM / 60.0   # Hz

    # Ambil semua baris per sumbu
    fx, mx = get_all_rows(df['fx_hz'], df['mx_mg'])
    fy, my = get_all_rows(df['fy_hz'], df['my_mg'])
    fz, mz = get_all_rows(df['fz_hz'], df['mz_mg'])

    # Nilai max untuk RMS & skala
    fx_max = df['fx_hz'].max(); mx_max = df['mx_mg'].max()
    fy_max = df['fy_hz'].max(); my_max = df['my_mg'].max()
    fz_max = df['fz_hz'].max(); mz_max = df['mz_mg'].max()

    vx = velocity_rms(fx_max, mx_max)
    vy = velocity_rms(fy_max, my_max)
    vz = velocity_rms(fz_max, mz_max)
    vw = max(vx, vy, vz)

    zone_all, col_all = iso_zone(vw)
    zone_x, _ = iso_zone(vx)
    zone_y, _ = iso_zone(vy)
    zone_z, _ = iso_zone(vz)

    # Batas frekuensi â€” gunakan all_f.max() agar SELURUH FREKUENSI (termasuk 45 Hz) 
    # terplot 100% utuh tanpa terpotong sumbu X
    all_f = np.concatenate([fx, fy, fz])
    if len(all_f) > 0:
        f_lim = float(np.ceil(max(all_f.max() * 1.10, 50.0)))
    else:
        f_lim = 60.0

    # Skala Y per sumbu (individual)
    yx     = float(np.ceil(max(mx_max, 5.0) * 1.35))
    yy     = float(np.ceil(max(my_max, 5.0) * 1.35))
    yz     = float(np.ceil(max(mz_max, 5.0) * 1.35))
    y_tk_x = max(1, int(np.ceil(yx / 6)))
    y_tk_y = max(1, int(np.ceil(yy / 6)))
    y_tk_z = max(1, int(np.ceil(yz / 6)))

    # â”€â”€ Baca Data RMS NYATA & Solusi Fluktuasi Per-Baris â”€â”€â”€â”€â”€â”€â”€â”€â”€
    rms_cols = [c for c in ['rms_mms', 'rms', 'v_rms', 'rms_total', 'rms_g'] if c in df.columns]
    use_row_calc = True

    if rms_cols:
        col_name = rms_cols[0]
        raw_rms  = df[col_name].values.astype(float)
        # Jika kolom di CSV memiliki nilai yang berbeda-beda (ada fluktuasi std > 0.01)
        if np.std(raw_rms) > 0.01:
            if col_name == 'rms_g':
                vw_trend = (raw_rms * 9810.0) / (2.0 * np.pi * 30.0) / np.sqrt(2.0)
            else:
                vw_trend = raw_rms
            use_row_calc = False
            print(f"[OK] Membaca fluktuasi RMS dari kolom '{col_name}' CSV ({len(vw_trend)} baris)")

    if use_row_calc:
        # Solusi: Hitung Velocity RMS dinamis per-baris dari mx, my, mz dan fx, fy, fz
        print("[INFO] Kolom RMS di CSV bernilai konstan/flat. Menghitung fluktuasi Velocity RMS nyata per-baris data...")
        vw_trend = np.zeros(n)
        for i in range(n):
            vx_row = velocity_rms(df['fx_hz'].iloc[i], df['mx_mg'].iloc[i])
            vy_row = velocity_rms(df['fy_hz'].iloc[i], df['my_mg'].iloc[i])
            vz_row = velocity_rms(df['fz_hz'].iloc[i], df['mz_mg'].iloc[i])
            vw_trend[i] = max(vx_row, vy_row, vz_row)

    # â”€â”€ Solusi Presisi 1:1 â€” Plot SETIAP BARIS CSV Tanpa Skipping â”€â”€
    # Setiap baris CSV diplot 1:1 tanpa ada yang melompat.
    trend = vw_trend

    t = np.arange(len(trend))
    vw_last = float(trend[-1]) if len(trend) > 0 else vw
    vw_max  = float(vw_trend.max()) if len(vw_trend) > 0 else vw
    vw_avg  = float(vw_trend.mean()) if len(vw_trend) > 0 else vw

    zone_all, col_all = iso_zone(vw_max)
    zone_x, _ = iso_zone(vx)
    zone_y, _ = iso_zone(vy)
    zone_z, _ = iso_zone(vz)

    # Diagnostik â€” verifikasi jumlah baris yang akan diplot
    print(f"[PLOT] Sumbu X: {len(fx):,} spike | freq {fx.min():.1f}-{fx.max():.1f} Hz | mag {mx.min():.0f}-{mx.max():.0f} mg")
    print(f"[PLOT] Sumbu Y: {len(fy):,} spike | freq {fy.min():.1f}-{fy.max():.1f} Hz | mag {my.min():.0f}-{my.max():.0f} mg")
    print(f"[PLOT] Sumbu Z: {len(fz):,} spike | freq {fz.min():.1f}-{fz.max():.1f} Hz | mag {mz.min():.0f}-{mz.max():.0f} mg")
    print(f"[PLOT] f_lim   : {f_lim:.1f} Hz  (persentil 99% dari semua frekuensi)")
    print(f"[STAT] RMS Last: {vw_last:.2f} mm/s | RMS Max: {vw_max:.2f} mm/s | RMS Avg: {vw_avg:.2f} mm/s")

    # â”€â”€ Figure â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    CX = '#0284C7'
    CY = '#D97706'
    CZ = '#DC2626'

    fig, axes = plt.subplots(2, 2, figsize=(14, 10.5), facecolor='white')
    fig.suptitle(
        f"ISO 10816-3  ({NOMINAL_RPM} RPM)  â€”  {zone_all}  |  MAX RMS: {vw_max:.2f} mm/s  (Avg: {vw_avg:.2f} mm/s)",
        fontsize=11.5, fontweight='bold', color=col_all, y=0.98
    )

    for ax in axes.flat:
        style_ax(ax)

    # â”€â”€ [1] Overall RMS Trend (Background Putih Tanpa Warna Block) â”€â”€
    ax0   = axes[0, 0]
    y_rms = max(5.0, float(np.ceil(vw_max * 1.25)))
    
    # Garis Putus-Putus Berwarna untuk Penanda Zona ISO 10816-3
    ax0.axhline(1.4, color='#16A34A', ls='--', lw=1.3, label='ZONE A Threshold (1.4 mm/s)')
    ax0.axhline(2.8, color='#D97706', ls='--', lw=1.3, label='ZONE B Threshold (2.8 mm/s)')
    ax0.axhline(4.5, color='#DC2626', ls='--', lw=1.3, label='ZONE C Threshold (4.5 mm/s)')
    
    # Solusi 1: Plot Tren Dinamis (Mengikuti Fluktuasi Setiap Sampel CSV)
    ax0.plot(t, trend, color='#0EA5E9', lw=1.8, marker='o', ms=3, 
             label=f'Trend Line (Last: {vw_last:.2f} | Max: {vw_max:.2f} | Avg: {vw_avg:.2f} mm/s)')
    
    # â”€â”€ HIGHLIGHT PUNCAK MAKSIMUM DENGAN BINTANG MERAH â”€â”€
    idx_max = np.argmax(trend)
    t_max_pt = t[idx_max]
    val_max_pt = trend[idx_max]
    
    ax0.plot(t_max_pt, val_max_pt, marker='*', color='#EF4444', ms=12, zorder=10,
             label=f'Max Peak ({val_max_pt:.2f} mm/s @ Sample #{t_max_pt})')
    
    ax0.annotate(
        f"MAX: {val_max_pt:.2f} mm/s\n(Sample #{t_max_pt})",
        xy=(t_max_pt, val_max_pt),
        xytext=(t_max_pt + len(t) * 0.05, val_max_pt - y_rms * 0.12),
        bbox=dict(boxstyle='round,pad=0.3', fc='#EF4444', ec='white', lw=1.3),
        arrowprops=dict(fc='#EF4444', ec='#EF4444', arrowstyle='->', lw=1.3),
        fontsize=8.0, fontweight='bold', color='white', zorder=11
    )

    ax0.annotate(
        f"Last: {vw_last:.2f} mm/s",
        xy=(t[-1], trend[-1]),
        xytext=(t[-1] - len(t) * 0.28, trend[-1] + y_rms * 0.08),
        bbox=dict(boxstyle='round,pad=0.3', fc='#0F172A', ec='#0EA5E9', lw=1.2),
        arrowprops=dict(fc='#0EA5E9', arrowstyle='->', lw=1.1),
        fontsize=8.0, fontweight='bold', color='white'
    )
    ax0.set_title(f"1. OVERALL VELOCITY RMS TREND  [Max: {vw_max:.2f} mm/s]", fontsize=10,
                  fontweight='bold', color='#0F172A', pad=7)
    ax0.set_xlabel("Time (Record Samples)", fontsize=9, fontweight='bold',
                   color='#0F172A', labelpad=3)
    ax0.set_ylabel("Velocity RMS (mm/s)", fontsize=9,
                   fontweight='bold', color='#0F172A')
    ax0.set_xlim(0, len(t) - 1)
    ax0.set_ylim(0, y_rms)
    ax0.legend(loc='upper left', fontsize=6.8, frameon=True,
               facecolor='white', edgecolor='#CBD5E1')

    # â”€â”€ [2] Spektrum Sumbu X â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ax1 = axes[0, 1]
    plot_spectrum_ax(
        ax1, fx, mx, CX, f_lim, yx,
        title=f"2. SPEKTRUM SUMBU X  â€”  {vx:.2f} mm/s  [{zone_x}]",
        ylabel=f"Magnitude (mg)   maks={mx_max:.0f} mg",
        y_tick=y_tk_x
    )
    label_peaks(ax1, fx, mx, f1x, CX, yx)

    # â”€â”€ [3] Spektrum Sumbu Y â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ax2 = axes[1, 0]
    plot_spectrum_ax(
        ax2, fy, my, CY, f_lim, yy,
        title=f"3. SPEKTRUM SUMBU Y  â€”  {vy:.2f} mm/s  [{zone_y}]",
        ylabel=f"Magnitude (mg)   maks={my_max:.0f} mg",
        y_tick=y_tk_y
    )
    label_peaks(ax2, fy, my, f1x, CY, yy)

    # â”€â”€ [4] Spektrum Sumbu Z â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ax3 = axes[1, 1]
    plot_spectrum_ax(
        ax3, fz, mz, CZ, f_lim, yz,
        title=f"4. SPEKTRUM SUMBU Z  â€”  {vz:.2f} mm/s  [{zone_z}]",
        ylabel=f"Magnitude (mg)   maks={mz_max:.0f} mg",
        y_tick=y_tk_z
    )
    label_peaks(ax3, fz, mz, f1x, CZ, yz)

    fig.subplots_adjust(left=0.07, right=0.97, top=0.92,
                        bottom=0.07, hspace=0.42, wspace=0.25)
    plt.show()


if __name__ == "__main__":
    main()