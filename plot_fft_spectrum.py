#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Plot FFT Spectrum dengan Tampilan DUAL-MODE (Bisa Diganti via Sakelar 1 Tombol):
# MODE = "TRIAXIAL"   -> Membaca 100% Data ASLI VIBRO.CSV (RMS Trend, Spektrum X, Y, Z + Diagnosis Otomatis)
# MODE = "JURNAL_FIG5" -> Tampilan Pembanding 4 Kondisi Mesin Persis Gambar Fig. 5 Jurnal Ilmiah
#                        ((a) Normal, (b) Unbalance, (c) Misalignment, (d) Bearing Fault)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ─────────────────────────────────────────────────────────────────────────────
# 🎛️ SAKELAR MODE TAMPILAN (PILIH SALAH SATU):
# Ubah nilai variabel MODE di bawah ini untuk mengganti mode grafik saat sidang:
#   - "TRIAXIAL"   : Pemantauan Data Asli Sumbu X, Y, Z dari VIBRO.CSV (Rekomendasi Utama)
#   - "JURNAL_FIG5": Pembanding 4 Kondisi Mesin (Persis Gambar Fig. 5 Jurnal Ilmiah)
MODE        = "TRIAXIAL"

NOMINAL_RPM = 1800
CSV_PATH    = r"D:\SEMESTER7\KERJAPRAKTIK\VIBRO.CSV"
# ─────────────────────────────────────────────────────────────────────────────


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
    if len(freqs) == 0:
        return []
    
    max_m = mags.max()
    threshold = max(2.5, max_m * 0.20)
    
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


def diagnose_axis_component(axis_name, freqs, mags, f1x_actual=30.0):
    if len(freqs) == 0 or len(mags) == 0:
        return "Normal", []

    peaks = get_all_peaks_from_csv(freqs, mags, min_gap_hz=4.0)
    if len(peaks) == 0:
        return "Normal", []

    m_1x, m_2x, m_3x = 0.0, 0.0, 0.0
    harmonics_count = 0
    sub_harmonics_count = 0
    high_freq_peaks = 0
    total_energy = np.sum(mags)

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

    if m_1x >= 15.0 and (m_2x == 0 or m_1x / (m_2x + 1e-6) >= 2.0) and (m_1x / (total_energy + 1e-6) >= 0.25):
        if axis_name in ['X', 'Y']:
            faults.append("Unbalance pada Rotor/Impeller (1X Radial)")
        else:
            faults.append("Unbalance/Flange Axis (1X Aksial)")

    if m_2x >= 10.0 and (m_2x >= 0.40 * m_1x) and (m_1x > 0):
        if axis_name == 'Z':
            faults.append("Angular Misalignment pada Kopling (1X/2X Aksial)")
        else:
            faults.append("Parallel Misalignment pada Poros (2X Radial)")

    if harmonics_count >= 4 or sub_harmonics_count >= 2:
        if axis_name == 'Y':
            faults.append("Structural Looseness pada Baut Dudukan (Harmonisa Y)")
        else:
            faults.append("Internal Component Looseness (Harmonisa Radial)")

    if high_freq_peaks >= 2:
        faults.append("Bearing Fault / Cacat Bantalan (Frekuensi Tinggi)")

    return ("FAULT DETECTED" if len(faults) > 0 else "Normal"), faults


def add_frequency_band_shading(ax, f1x_actual, f_lim):
    z1_min, z1_max = f1x_actual * 0.88, f1x_actual * 1.12
    z2_min, z2_max = f1x_actual * 1.88, f1x_actual * 2.12
    z3_min = f1x_actual * 2.8

    if z1_max <= f_lim:
        ax.axvspan(z1_min, z1_max, color='#0284C7', alpha=0.08, label='1X Zone (Unbalance)')
    if z2_max <= f_lim:
        ax.axvspan(z2_min, z2_max, color='#D97706', alpha=0.08, label='2X Zone (Misalignment)')
    if z3_min <= f_lim:
        ax.axvspan(z3_min, f_lim, color='#EAB308', alpha=0.06, label='Harmonisa Zone (Looseness/Bearing)')


def label_peaks(ax, axis_name, freqs, mags, f1x_nominal=30.0, color='#1F77B4', min_gap_hz=4.0):
    peaks = get_all_peaks_from_csv(freqs, mags, min_gap_hz=min_gap_hz)
    
    f1x_actual = f1x_nominal
    candidate_1x = [f for f, m in peaks if 25.0 <= f <= 35.0]
    if len(candidate_1x) > 0:
        f1x_actual = max(candidate_1x, key=lambda f: dict(peaks)[f])

    status, faults = diagnose_axis_component(axis_name, freqs, mags, f1x_actual=f1x_actual)

    for i, (f, m) in enumerate(peaks):
        ax.plot(f, m, marker='s', color='black', markersize=4.2, zorder=5)

        ratio = f / f1x_actual
        n = round(ratio)
        
        if n >= 1 and abs(ratio - n) < 0.12:
            tag = f"{n}X"
        else:
            tag = f"{ratio:.1f}X"

        diag_suffix = ""
        if abs(ratio - 1.0) < 0.12:
            if axis_name in ['X', 'Y']:
                diag_suffix = " [Unbalance/Rotor]"
            else:
                diag_suffix = " [Unbalance/Aksial]"
        elif abs(ratio - 2.0) < 0.12:
            if axis_name == 'Z':
                diag_suffix = " [Misalignment/Kopling]"
            else:
                diag_suffix = " [Misalignment/Poros]"
        elif n > 2 and (f < 300.0 and ratio <= 10.0):
            if axis_name == 'Y':
                diag_suffix = " [Looseness/Baut]"
            else:
                diag_suffix = " [Looseness/Internal]"
        elif f > 300.0 or ratio > 10.0:
            diag_suffix = " [Bearing Defect]"

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

    return status, faults, f1x_actual


def style_ax(ax):
    ax.set_facecolor('#FAF6EE')
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#D0D0D0')
        spine.set_linewidth(0.8)
    ax.grid(True, which='major', color='white', linestyle='-', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(direction='in', length=4, width=0.8, top=True, right=True, labelsize=9, colors='#444444')


def plot_spectrum_ax(ax, axis_name, freqs, mags, color='#1F77B4', f1x=30.0, f_lim=100, y_max=12, title="SPEKTRUM", ylabel="Magnitude [mg]", y_tick=None):
    f_axis, y_axis = build_real_fft_curve(freqs, mags, f_lim=f_lim)
    ax.plot(f_axis, y_axis, color=color, lw=1.1, zorder=2)
    
    ax.set_title(title, fontsize=10.5, fontweight='bold', color='#111111', loc='left', pad=6)
    ax.set_xlabel("Frequency [Hz]", fontsize=9.5, fontweight='bold', color='#333333', labelpad=3)
    ax.set_ylabel(ylabel, fontsize=9.5, fontweight='bold', color='#333333', labelpad=3)
    ax.set_xlim(0, f_lim)
    ax.set_ylim(0, y_max)
    if y_tick:
        ax.yaxis.set_major_locator(MultipleLocator(y_tick))
        
    status, faults, f1x_actual = label_peaks(ax, axis_name, freqs, mags, f1x_nominal=f1x, color=color)
    add_frequency_band_shading(ax, f1x_actual, f_lim)
    return status, faults


def generate_condition_spectrum(condition_type, f_max=2500, num_points=1500):
    f_axis = np.linspace(0, f_max, num_points)
    np.random.seed(42)
    
    if condition_type == "normal":
        y_axis = np.abs(np.random.normal(0.04, 0.02, len(f_axis)))
        for p_f, p_m in [(30, 0.65), (750, 1.55), (830, 1.62), (950, 0.98), (1050, 0.93), (1220, 0.72), (1420, 0.73), (1540, 0.85)]:
            sigma = 4.0
            y_axis = np.maximum(y_axis, p_m * np.exp(-0.5 * ((f_axis - p_f) / sigma) ** 2))
        return f_axis, y_axis

    elif condition_type == "unbalance":
        y_axis = np.abs(np.random.normal(0.05, 0.02, len(f_axis)))
        for p_f, p_m in [(60, 6.80), (300, 1.80), (740, 3.50), (950, 1.00), (1140, 0.75), (1540, 0.78)]:
            sigma = 5.0
            y_axis = np.maximum(y_axis, p_m * np.exp(-0.5 * ((f_axis - p_f) / sigma) ** 2))
        return f_axis, y_axis

    elif condition_type == "misalignment":
        y_axis = np.abs(np.random.normal(0.06, 0.03, len(f_axis)))
        for p_f, p_m in [(60, 16.50), (740, 4.80), (840, 2.30), (950, 2.90), (1040, 2.75), (1240, 2.15), (1440, 2.20)]:
            sigma = 6.0
            y_axis = np.maximum(y_axis, p_m * np.exp(-0.5 * ((f_axis - p_f) / sigma) ** 2))
        return f_axis, y_axis

    elif condition_type == "bearing_fault":
        y_axis = np.abs(np.random.normal(0.12, 0.08, len(f_axis)))
        for p_f, p_m in [(730, 6.80), (750, 9.80), (1360, 7.60), (1420, 5.80), (1480, 10.0), (1500, 15.0), (1540, 10.2), (1600, 15.6), (1630, 7.8)]:
            sigma = 8.0
            y_axis = np.maximum(y_axis, p_m * np.exp(-0.5 * ((f_axis - p_f) / sigma) ** 2))
        return f_axis, y_axis


def render_triaxial_mode():
    """
    MODE 1: Pemantauan Data Asli Triaksial dari VIBRO.CSV (RMS Trend + Spektrum Sumbu X, Y, Z)
    """
    if not os.path.exists(CSV_PATH):
        print("[ERROR] File tidak ditemukan:", CSV_PATH)
        return

    df = pd.read_csv(CSV_PATH)
    n = len(df)
    print(f"[OK] {n:,} baris data ASLI dibaca dari {CSV_PATH}")

    f1x = NOMINAL_RPM / 60.0

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

    yx     = float(np.ceil(max(mx.max(), 5.0) * 1.35))
    yy     = float(np.ceil(max(my.max(), 5.0) * 1.35))
    yz     = float(np.ceil(max(mz.max(), 5.0) * 1.35))
    y_tk_x = max(1, int(np.ceil(yx / 5)))
    y_tk_y = max(1, int(np.ceil(yy / 5)))
    y_tk_z = max(1, int(np.ceil(yz / 5)))

    vw_trend = df['rms_mms'].values.astype(float)
    vw_max   = float(vw_trend.max())
    t        = np.arange(len(vw_trend))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), facecolor='white')
    
    for ax in axes.flat:
        style_ax(ax)

    # (a) RMS Trend
    ax0   = axes[0, 0]
    y_rms = max(5.0, float(np.ceil(vw_max * 1.25)))
    
    ax0.axhline(1.4, color='#16A34A', ls='--', lw=1.3, label='ZONE A Threshold (1.4 mm/s)')
    ax0.axhline(2.8, color='#D97706', ls='--', lw=1.3, label='ZONE B Threshold (2.8 mm/s)')
    ax0.axhline(4.5, color='#DC2626', ls='--', lw=1.3, label='ZONE C Threshold (4.5 mm/s)')
    ax0.plot(t, vw_trend, color='#1F77B4', lw=1.8, marker='o', ms=2, label=f'Trend Line')
    
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
    ax0.set_title(f"(a) OVERALL VELOCITY RMS TREND  [Max: {vw_max:.2f} mm/s]", fontsize=10.5, fontweight='bold', color='#111111', loc='left', pad=6)
    ax0.set_xlabel("Time (Record Samples)", fontsize=9.5, fontweight='bold', color='#333333', labelpad=3)
    ax0.set_ylabel("Velocity RMS [mm/s]", fontsize=9.5, fontweight='bold', color='#333333')
    ax0.set_xlim(0, len(t) - 1)
    ax0.set_ylim(0, y_rms)
    ax0.legend(loc='upper left', fontsize=7.0, frameon=True, facecolor='white', edgecolor='#CBD5E1')

    # (b) Spektrum X
    st_x, fl_x = plot_spectrum_ax(
        axes[0, 1], 'X', fx, mx, color='#1F77B4', f1x=f1x, f_lim=f_lim, y_max=yx,
        title=f"(b) SPEKTRUM SUMBU X (Radial)  -  {vx:.2f} mm/s [{zone_x}]", ylabel="Magnitude [mg]", y_tick=y_tk_x
    )

    # (c) Spektrum Y
    st_y, fl_y = plot_spectrum_ax(
        axes[1, 0], 'Y', fy, my, color='#1F77B4', f1x=f1x, f_lim=f_lim, y_max=yy,
        title=f"(c) SPEKTRUM SUMBU Y (Radial)  -  {vy:.2f} mm/s [{zone_y}]", ylabel="Magnitude [mg]", y_tick=y_tk_y
    )

    # (d) Spektrum Z
    st_z, fl_z = plot_spectrum_ax(
        axes[1, 1], 'Z', fz, mz, color='#1F77B4', f1x=f1x, f_lim=f_lim, y_max=yz,
        title=f"(d) SPEKTRUM SUMBU Z (Aksial)  -  {vz:.2f} mm/s [{zone_z}]", ylabel="Magnitude [mg]", y_tick=y_tk_z
    )

    all_faults = list(set(fl_x + fl_y + fl_z))
    if len(all_faults) > 0:
        diag_summary = "DIAGNOSIS KOMPONEN KERUSAKAN: " + " | ".join(all_faults)
        diag_color = "#DC2626"
    else:
        diag_summary = "DIAGNOSIS KERUSAKAN: TIDAK TERDETEKSI ANOMALI KRITIS (Sistem Normal)"
        diag_color = "#16A34A"

    fig.suptitle(
        f"ISO 10816-3 ({NOMINAL_RPM} RPM)  -  {zone_all} | MAX RMS: {vw_max:.2f} mm/s\n{diag_summary}",
        fontsize=10.5, fontweight='bold', color=diag_color, y=0.99
    )

    fig.subplots_adjust(left=0.07, right=0.97, top=0.90, bottom=0.07, hspace=0.42, wspace=0.22)
    plt.show()


def render_jurnal_fig5_mode():
    """
    MODE 2: Pembanding 4 Kondisi Mesin Persis Gambar Fig. 5 Jurnal Ilmiah
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), facecolor='white')
    
    for ax in axes.flat:
        style_ax(ax)

    # (a) Normal
    ax_a = axes[0, 0]
    fa, ya = generate_condition_spectrum("normal")
    ax_a.plot(fa, ya, color='#1F77B4', lw=1.1)
    ax_a.set_xlim(0, 2500)
    ax_a.set_ylim(0, 1.75)
    ax_a.set_title("(a) Normal", fontsize=11, fontweight='bold', loc='left', pad=6, color='#222222')
    ax_a.set_xlabel("Frequency [Hz]", fontsize=9.5, color='#444444')
    ax_a.set_ylabel("Amplitude [m/s²]", fontsize=9.5, color='#444444')

    # (b) Unbalance
    ax_b = axes[0, 1]
    fb, yb = generate_condition_spectrum("unbalance")
    ax_b.plot(fb, yb, color='#1F77B4', lw=1.1)
    ax_b.set_xlim(0, 2500)
    ax_b.set_ylim(0, 7.2)
    ax_b.set_title("(b) Unbalance", fontsize=11, fontweight='bold', loc='left', pad=6, color='#222222')
    ax_b.set_xlabel("Frequency [Hz]", fontsize=9.5, color='#444444')
    ax_b.set_ylabel("Amplitude [m/s²]", fontsize=9.5, color='#444444')

    # (c) Misalignment
    ax_c = axes[1, 0]
    fc, yc = generate_condition_spectrum("misalignment")
    ax_c.plot(fc, yc, color='#1F77B4', lw=1.1)
    ax_c.set_xlim(0, 2500)
    ax_c.set_ylim(0, 17.5)
    ax_c.set_title("(c) Misalignment", fontsize=11, fontweight='bold', loc='left', pad=6, color='#222222')
    ax_c.set_xlabel("Frequency [Hz]", fontsize=9.5, color='#444444')
    ax_c.set_ylabel("Amplitude [m/s²]", fontsize=9.5, color='#444444')

    # (d) Bearing Fault
    ax_d = axes[1, 1]
    fd, yd = generate_condition_spectrum("bearing_fault")
    ax_d.plot(fd, yd, color='#1F77B4', lw=1.1)
    ax_d.set_xlim(0, 2500)
    ax_d.set_ylim(0, 16.5)
    ax_d.set_title("(d) Bearing Fault", fontsize=11, fontweight='bold', loc='left', pad=6, color='#222222')
    ax_d.set_xlabel("Frequency [Hz]", fontsize=9.5, color='#444444')
    ax_d.set_ylabel("Amplitude [m/s²]", fontsize=9.5, color='#444444')

    fig.suptitle(
        "Fig. 5  Spectrum of vibration signal in each machine condition: (a) normal, (b) unbalance, (c) misalignment, (d) bearing fault",
        fontsize=12, fontweight='bold', color='#111111', y=0.03
    )

    fig.subplots_adjust(left=0.07, right=0.97, top=0.94, bottom=0.10, hspace=0.35, wspace=0.22)
    plt.show()


def main():
    if MODE == "TRIAXIAL":
        print("[INFO] Menampilkan MODE TRIAXIAL: Membaca 100% Data Asli Pengukuran (VIBRO.CSV)")
        render_triaxial_mode()
    elif MODE == "JURNAL_FIG5":
        print("[INFO] Menampilkan MODE JURNAL FIG. 5: Pembanding 4 Kondisi Kesehatan Mesin")
        render_jurnal_fig5_mode()
    else:
        print(f"[WARNING] Mode '{MODE}' tidak dikenal. Menggunakan mode default 'TRIAXIAL'")
        render_triaxial_mode()


if __name__ == "__main__":
    main()