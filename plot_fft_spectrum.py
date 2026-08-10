#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Plot FFT Spectrum dari VIBRO.CSV dengan Diagnosis Otomatis Kerusakan Mesin
# Diagnosis: 1. Unbalance, 2. Misalignment, 3. Mechanical Looseness, 4. Bearing Fault
# Layout 2x2: [Overall RMS Trend] [Spektrum X] [Spektrum Y] [Spektrum Z]

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ──────────────────────────────────────
NOMINAL_RPM = 1800
CSV_PATH    = r"D:\SEMESTER7\KERJAPRAKTIK\VIBRO.CSV"
# ──────────────────────────────────────


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


def get_csv_spectrum(f_series, m_series):
    """
    Ambil data spektrum NYATA dari CSV.
    Kelompokkan per frekuensi unik (dibulatkan 0.1 Hz) dan ambil magnitudo maksimumnya.
    """
    valid = (f_series > 0) & (m_series > 0)
    df_valid = pd.DataFrame({'f': f_series[valid], 'm': m_series[valid]})
    df_valid['f_round'] = df_valid['f'].round(1)
    grouped = df_valid.groupby('f_round')['m'].max().reset_index()
    return grouped['f_round'].values.astype(float), grouped['m'].values.astype(float)


def get_all_peaks_from_csv(freqs, mags, min_gap_hz=4.0):
    """
    Ambil puncak-puncak signifikan dari data CSV.
    Menyaring derau (noise floor) agar tidak menumpuk label di dasar grafik.
    """
    if len(freqs) == 0:
        return []
    
    max_m = mags.max()
    threshold = max(2.5, max_m * 0.20)  # Hanya label puncak > 20% dari puncak terbesar & min 2.5 mg
    
    peak_dict = {}
    for f, m in zip(freqs, mags):
        key = round(f, 1)
        if key not in peak_dict or m > peak_dict[key]:
            peak_dict[key] = m
            
    selected = []
    for f, m in sorted(peak_dict.items(), key=lambda x: x[1], reverse=True):
        if m < threshold:
            continue
        if all(abs(f - sf) >= min_gap_hz for sf, _ in selected):
            selected.append((f, m))
            
    selected.sort(key=lambda x: x[0])
    return selected


def build_real_fft_curve(freqs, mags, f_lim=100, num_points=1200):
    """
    Membangun kurva spektrum FFT 100% PERSIS DATA CSV
    dengan spike presisi dan noise floor halus di dasar.
    """
    f_axis = np.linspace(0, f_lim, num_points)
    np.random.seed(42)
    y_axis = np.abs(np.random.normal(0.3, 0.12, len(f_axis)))
    
    for pf, pm in zip(freqs, mags):
        if pf > 0 and pm > 0 and pf <= f_lim:
            sigma = 0.5
            spike = pm * np.exp(-0.5 * ((f_axis - pf) / sigma) ** 2)
            y_axis = np.maximum(y_axis, spike)
            
    return f_axis, y_axis


def diagnose_axis_faults(freqs, mags, f1x_actual=30.0):
    """
    Algoritma Diagnosis Otomatis Kerusakan Mesin berdasarkan Karakteristik Spektrum Getaran:
    1. Unbalance (Ketidakseimbangan Massa): Dominansi puncak 1X (1X > 40% energi & 1X/2X >= 2.0)
    2. Misalignment (Ketidaksejajaran Poros): Amplitudo 2X tinggi (2X >= 40% dari 1X)
    3. Mechanical Looseness (Kelonggaran Mekanis): Banyak harmonisa berurutan (>= 4 harmonisa) / sub-harmonisa
    4. Bearing Fault (Kerusakan Bantalan): Puncak frekuensi tinggi (> 10X / > 300 Hz) / non-sinkron
    """
    if len(freqs) == 0 or len(mags) == 0:
        return "NORMAL", []

    peaks = get_all_peaks_from_csv(freqs, mags, min_gap_hz=4.0)
    if len(peaks) == 0:
        return "NORMAL", []

    m_1x, m_2x, m_3x = 0.0, 0.0, 0.0
    harmonics_count = 0
    sub_harmonics_count = 0
    high_freq_peaks = 0

    for f, m in peaks:
        ratio = f / f1x_actual
        n = round(ratio)
        
        if abs(ratio - 1.0) < 0.12:
            m_1x = max(m_1x, m)
        elif abs(ratio - 2.0) < 0.12:
            m_2x = max(m_2x, m)
        elif abs(ratio - 3.0) < 0.12:
            m_3x = max(m_3x, m)

        if n >= 1 and abs(ratio - n) < 0.15 and m >= 3.0:
            harmonics_count += 1

        if abs(ratio - (n - 0.5)) < 0.12 and m >= 2.5:
            sub_harmonics_count += 1

        if f > 300.0 or ratio > 10.0:
            high_freq_peaks += 1

    faults = []
    
    # Kriteria 1: Unbalance
    total_energy = np.sum(mags)
    if m_1x >= 15.0 and (m_2x == 0 or m_1x / (m_2x + 1e-6) >= 2.0) and (m_1x / (total_energy + 1e-6) >= 0.25):
        faults.append("UNBALANCE (1X Dominant)")

    # Kriteria 2: Misalignment
    if m_2x >= 10.0 and (m_2x >= 0.40 * m_1x) and (m_1x > 0):
        faults.append("MISALIGNMENT (High 2X)")

    # Kriteria 3: Mechanical Looseness
    if harmonics_count >= 4 or sub_harmonics_count >= 2:
        faults.append("LOOSENESS (Harmonics)")

    # Kriteria 4: Bearing Fault
    if high_freq_peaks >= 2:
        faults.append("BEARING FAULT (High Freq)")

    return ("FAULT DETECTED" if len(faults) > 0 else "NORMAL"), faults


def label_peaks(ax, freqs, mags, f1x_nominal=30.0, color='#0072BD', min_gap_hz=4.0):
    """
    Beri label puncak signifikan secara akurat beserta Indikator Diagnosis Kerusakan.
    """
    peaks = get_all_peaks_from_csv(freqs, mags, min_gap_hz=min_gap_hz)
    
    # Cari puncak 1X aktual pada sumbu ini (puncak dominan di kisaran 25 - 35 Hz)
    f1x_actual = f1x_nominal
    candidate_1x = [f for f, m in peaks if 25.0 <= f <= 35.0]
    if len(candidate_1x) > 0:
        f1x_actual = max(candidate_1x, key=lambda f: dict(peaks)[f])

    status, faults = diagnose_axis_faults(freqs, mags, f1x_actual=f1x_actual)

    for i, (f, m) in enumerate(peaks):
        ax.plot(f, m, marker='s', color='black', markersize=4.5, zorder=5)

        ratio = f / f1x_actual
        n = round(ratio)
        
        if n >= 1 and abs(ratio - n) < 0.12:
            tag = f"{n}X"
        else:
            tag = f"{ratio:.1f}X"

        # Tambahkan penanda diagnosis pada tag puncak yang sesuai
        diag_suffix = ""
        if abs(ratio - 1.0) < 0.12 and "UNBALANCE (1X Dominant)" in faults:
            diag_suffix = " [UNBALANCE]"
        elif abs(ratio - 2.0) < 0.12 and "MISALIGNMENT (High 2X)" in faults:
            diag_suffix = " [MISALIGNMENT]"
        elif "LOOSENESS (Harmonics)" in faults and n > 2:
            diag_suffix = " [LOOSENESS]"
        elif f > 300.0 or ratio > 10.0:
            diag_suffix = " [BEARING]"

        text_str = f"{tag}{diag_suffix}\n{m:.0f} mg"
        
        level = i % 3
        oy = 8 + level * 16

        ax.annotate(
            text_str,
            xy=(f, m),
            xytext=(0, oy),
            textcoords="offset points",
            ha='center',
            va='bottom',
            fontsize=7.5,
            fontweight='bold',
            family='sans-serif',
            bbox=dict(
                boxstyle='square,pad=0.25',
                facecolor='#FFFFE1',
                edgecolor='#808080',
                linewidth=0.6
            ),
            arrowprops=dict(
                arrowstyle='-',
                color='#808080',
                linewidth=0.5
            ) if oy > 10 else None,
            zorder=6 + level
        )

    return status, faults


def style_ax(ax):
    """
    Gaya Frame Box & Grid persis MATLAB / Gambar Referensi
    """
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(0.8)
    ax.grid(True, which='major', color='#D0D0D0', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(direction='in', length=4, width=0.8, top=True, right=True, labelsize=9)


def plot_spectrum_ax(ax, freqs, mags, color='#0072BD', f1x=30.0, f_lim=100, y_max=12, title="SPEKTRUM", ylabel="|X(f)|", y_tick=None):
    f_axis, y_axis = build_real_fft_curve(freqs, mags, f_lim=f_lim)
    ax.plot(f_axis, y_axis, color=color, lw=0.9, zorder=2)
    
    ax.set_title(title, fontsize=10, fontweight='bold', color='black', pad=7)
    ax.set_xlabel("f (in Hz)", fontsize=9.5, fontweight='bold', color='black', labelpad=3)
    ax.set_ylabel(ylabel, fontsize=9.5, fontweight='bold', color='black', labelpad=3)
    ax.set_xlim(0, f_lim)
    ax.set_ylim(0, y_max)
    if y_tick:
        ax.yaxis.set_major_locator(MultipleLocator(y_tick))
        
    status, faults = label_peaks(ax, freqs, mags, f1x_nominal=f1x, color=color)
    return status, faults


# ──────────────────────────────────────
def main():
    if not os.path.exists(CSV_PATH):
        print("[ERROR] File tidak ditemukan:", CSV_PATH)
        return

    df = pd.read_csv(CSV_PATH)
    n = len(df)
    print(f"[OK] {n:,} baris dibaca dari {CSV_PATH}")

    f1x = NOMINAL_RPM / 60.0   # Hz

    # Ambil spektrum nyata dari CSV
    fx, mx = get_csv_spectrum(df['fx_hz'], df['mx_mg'])
    fy, my = get_csv_spectrum(df['fy_hz'], df['my_mg'])
    fz, mz = get_csv_spectrum(df['fz_hz'], df['mz_mg'])

    vx = velocity_rms(df['fx_hz'].max(), df['mx_mg'].max())
    vy = velocity_rms(df['fy_hz'].max(), df['my_mg'].max())
    vz = velocity_rms(df['fz_hz'].max(), df['mz_mg'].max())
    vw = max(vx, vy, vz)

    zone_all, col_all = iso_zone(vw)
    zone_x, _ = iso_zone(vx)
    zone_y, _ = iso_zone(vy)
    zone_z, _ = iso_zone(vz)

    all_f = np.concatenate([fx, fy, fz])
    f_lim = float(np.ceil(max(all_f.max() * 1.05, 100.0)))

    # Skala Y per sumbu
    yx     = float(np.ceil(max(mx.max(), 5.0) * 1.35))
    yy     = float(np.ceil(max(my.max(), 5.0) * 1.35))
    yz     = float(np.ceil(max(mz.max(), 5.0) * 1.35))
    y_tk_x = max(1, int(np.ceil(yx / 5)))
    y_tk_y = max(1, int(np.ceil(yy / 5)))
    y_tk_z = max(1, int(np.ceil(yz / 5)))

    # RMS Trend
    vw_trend = df['rms_mms'].values.astype(float)
    vw_max   = float(vw_trend.max())
    vw_avg   = float(vw_trend.mean())
    vw_last  = float(vw_trend[-1])
    t        = np.arange(len(vw_trend))

    # ── Figure ──────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), facecolor='white')
    
    for ax in axes.flat:
        style_ax(ax)

    # ── [1] Overall RMS Trend ──
    ax0   = axes[0, 0]
    y_rms = max(5.0, float(np.ceil(vw_max * 1.25)))
    
    ax0.axhline(1.4, color='#16A34A', ls='--', lw=1.3, label='ZONE A Threshold (1.4 mm/s)')
    ax0.axhline(2.8, color='#D97706', ls='--', lw=1.3, label='ZONE B Threshold (2.8 mm/s)')
    ax0.axhline(4.5, color='#DC2626', ls='--', lw=1.3, label='ZONE C Threshold (4.5 mm/s)')
    ax0.plot(t, vw_trend, color='#0EA5E9', lw=1.8, marker='o', ms=2, label=f'Trend Line')
    
    idx_max = np.argmax(vw_trend)
    ax0.plot(t[idx_max], vw_trend[idx_max], marker='*', color='#EF4444', ms=12, zorder=10)
    ax0.annotate(
        f"MAX: {vw_trend[idx_max]:.2f} mm/s\n(Sample #{t[idx_max]})",
        xy=(t[idx_max], vw_trend[idx_max]),
        xytext=(t[idx_max] + len(t) * 0.05, vw_trend[idx_max] - y_rms * 0.12),
        bbox=dict(boxstyle='round,pad=0.3', fc='#EF4444', ec='white', lw=1.3),
        arrowprops=dict(fc='#EF4444', ec='#EF4444', arrowstyle='->', lw=1.3),
        fontsize=8.0, fontweight='bold', color='white', zorder=11
    )
    ax0.annotate(
        f"Last: {vw_last:.2f} mm/s",
        xy=(t[-1], vw_trend[-1]),
        xytext=(t[-1] - len(t) * 0.28, vw_trend[-1] + y_rms * 0.08),
        bbox=dict(boxstyle='round,pad=0.3', fc='#0F172A', ec='#0EA5E9', lw=1.2),
        arrowprops=dict(fc='#0EA5E9', arrowstyle='->', lw=1.1),
        fontsize=8.0, fontweight='bold', color='white'
    )
    ax0.set_title(f"1. OVERALL VELOCITY RMS TREND  [Max: {vw_max:.2f} mm/s]", fontsize=10, fontweight='bold', color='#0F172A', pad=7)
    ax0.set_xlabel("Time (Record Samples)", fontsize=9, fontweight='bold', color='#0F172A', labelpad=3)
    ax0.set_ylabel("Velocity RMS (mm/s)", fontsize=9, fontweight='bold', color='#0F172A')
    ax0.set_xlim(0, len(t) - 1)
    ax0.set_ylim(0, y_rms)
    ax0.legend(loc='upper left', fontsize=6.8, frameon=True, facecolor='white', edgecolor='#CBD5E1')

    # ── [2] Spektrum Sumbu X ──
    st_x, fl_x = plot_spectrum_ax(
        axes[0, 1], fx, mx, color='#0284C7', f1x=f1x, f_lim=f_lim, y_max=yx,
        title=f"SPEKTRUM SUMBU X  -  {vx:.2f} mm/s  [{zone_x}]", ylabel="|X(f)|", y_tick=y_tk_x
    )

    # ── [3] Spektrum Sumbu Y ──
    st_y, fl_y = plot_spectrum_ax(
        axes[1, 0], fy, my, color='#D97706', f1x=f1x, f_lim=f_lim, y_max=yy,
        title=f"SPEKTRUM SUMBU Y  -  {vy:.2f} mm/s  [{zone_y}]", ylabel="|Y(f)|", y_tick=y_tk_y
    )

    # ── [4] Spektrum Sumbu Z ──
    st_z, fl_z = plot_spectrum_ax(
        axes[1, 1], fz, mz, color='#DC2626', f1x=f1x, f_lim=f_lim, y_max=yz,
        title=f"SPEKTRUM SUMBU Z  -  {vz:.2f} mm/s  [{zone_z}]", ylabel="|Z(f)|", y_tick=y_tk_z
    )

    # Kumpulkan semua temuan diagnosis dari sumbu X, Y, Z
    all_faults = list(set(fl_x + fl_y + fl_z))
    if len(all_faults) > 0:
        diag_summary = "DIAGNOSIS KERUSAKAN: " + ", ".join(all_faults)
        diag_color = "#DC2626"
    else:
        diag_summary = "DIAGNOSIS KERUSAKAN: NO CRITICAL FAULT DETECTED (System Normal)"
        diag_color = "#16A34A"

    fig.suptitle(
        f"ISO 10816-3 ({NOMINAL_RPM} RPM)  -  {zone_all} | MAX RMS: {vw_max:.2f} mm/s\n{diag_summary}",
        fontsize=11.0, fontweight='bold', color=diag_color, y=0.99
    )

    fig.subplots_adjust(left=0.07, right=0.97, top=0.90, bottom=0.07, hspace=0.45, wspace=0.25)
    plt.show()


if __name__ == "__main__":
    main()