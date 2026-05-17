#!/usr/bin/env python3
"""
analyse_biomecanique.py — Pipeline d'analyse biomécanique vidéo complet
Basé sur optical_flow_version_plus.ipynb

Usage:
    python analyse_biomecanique.py <video_path> [options]

Options:
    --output-dir  DIR     Dossier de sortie (défaut: outputs_bio/)
    --width  INT          Largeur de travail (défaut: 640)
    --height INT          Hauteur de travail (défaut: 360)
    --tracker STR         KCF ou CSRT (défaut: CSRT)
    --roi X Y W H         ROI manuelle (sinon = centre auto)
    --n-rois  INT         Nombre de ROIs pour l'analyse multi-ROI (défaut: 2)
    --redetect INT        Re-détecter les points LK toutes les N frames (défaut: 15)
    --target-diff FLOAT   Seuil diff L1 pour stride adaptatif (défaut: 3.0)
    --no-lk               Désactiver Lucas-Kanade
    --no-heatmap          Désactiver la heatmap accumulée
    --no-bgs              Désactiver le background subtraction
    --no-multiroi         Désactiver l'analyse multi-ROI
    --no-dashboard        Désactiver le dashboard final
    --no-display          Pas de fenêtre cv2.imshow (headless/serveur)

Sorties dans --output-dir :
    resultat_tracking.mp4     Vidéo annotée : tracker + vecteurs flux + signal + reps
    resultat_couleurs.mp4     Side-by-side : original | Farneback HSV
    resultat_bgs.mp4          Side-by-side : trajectoire centroïde | foreground mask
    heatmap_mouvement.png     Heatmap du mouvement cumulé
    dashboard_final.png       Dashboard récapitulatif de toutes les métriques
    optical_flow_lk.csv       Signal Lucas-Kanade (frame, mag, dx, dy)
    frames_output.csv         Index des frames analysées
"""

import sys, os, argparse, csv
from pathlib import Path
from itertools import combinations

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import savgol_filter, find_peaks

# ─────────────────────────────────────────────────────────────────────────────
# Arguments
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Analyse biomécanique vidéo (Farneback + LK + Tracker + BGS + Dashboard)"
    )
    p.add_argument("video",          help="Chemin vers la vidéo source")
    p.add_argument("--output-dir",   default="outputs_bio")
    p.add_argument("--width",        type=int,   default=640)
    p.add_argument("--height",       type=int,   default=360)
    p.add_argument("--tracker",      default="CSRT", choices=["KCF", "CSRT"])
    p.add_argument("--roi",          type=int,   nargs=4, metavar=("X","Y","W","H"),
                   help="ROI principale manuelle (pixels)")
    p.add_argument("--n-rois",       type=int,   default=2,
                   help="Nombre de ROIs pour l'analyse multi-ROI (défaut: 2)")
    p.add_argument("--redetect",     type=int,   default=15,
                   help="Re-détection LK toutes les N frames (défaut: 15)")
    p.add_argument("--target-diff",  type=float, default=3.0)
    p.add_argument("--no-lk",        action="store_true")
    p.add_argument("--no-heatmap",   action="store_true")
    p.add_argument("--no-bgs",       action="store_true")
    p.add_argument("--no-multiroi",  action="store_true")
    p.add_argument("--no-dashboard", action="store_true")
    p.add_argument("--no-display",   action="store_true",
                   help="Désactiver cv2.imshow (mode headless/serveur)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires
# ─────────────────────────────────────────────────────────────────────────────

def put_label(frame, text, pos, scale=0.6, color=(255, 255, 255), thickness=1):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

def make_writer(path, fps, w, h):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    assert writer.isOpened(), f"Impossible de créer la vidéo : {path}"
    return writer

def smooth(signal, wl=11, poly=2):
    n = len(signal)
    wl = min(wl, n if n % 2 == 1 else max(3, n - 1))
    if wl < 3:
        return signal.copy()
    return savgol_filter(signal, window_length=wl, polyorder=poly)

def auto_roi(frame_w, frame_h, ratio=0.25):
    rw = int(frame_w * ratio)
    rh = int(frame_h * ratio)
    rx = (frame_w - rw) // 2
    ry = (frame_h - rh) // 2
    return (rx, ry, rw, rh)

def flow_to_hsv_bgr(flow: np.ndarray) -> np.ndarray:
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# ─────────────────────────────────────────────────────────────────────────────
# Optical Flow CPU Farneback
# ─────────────────────────────────────────────────────────────────────────────

def build_farneback():
    """
    Retourne une fonction compute_flow(prev_gray, gray) → flow (H,W,2) sur CPU.
    """
    def compute_flow_cpu(prev_gray, gray):
        return cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
    return compute_flow_cpu


# ─────────────────────────────────────────────────────────────────────────────
# Chargement vidéo
# ─────────────────────────────────────────────────────────────────────────────

def open_video(video_path, frame_w, frame_h):
    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), f"Impossible d'ouvrir : {video_path}"
    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_nat        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_nat        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Résolution native  : {w_nat}x{h_nat}")
    print(f"  FPS : {fps:.1f} | Frames : {total_frames} | Durée : {total_frames/fps:.1f}s")
    return cap, fps, total_frames


def read_first_frame(video_path, frame_w, frame_h):
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    assert ret
    frame = cv2.resize(frame, (frame_w, frame_h))
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Tracker OpenCV
# ─────────────────────────────────────────────────────────────────────────────

def make_tracker(tracker_type: str):
    """
    Crée le tracker demandé, avec fallback automatique vers cv2.legacy 
    pour les versions récentes d'OpenCV.
    """
    if tracker_type == "KCF":
        if hasattr(cv2, 'TrackerKCF_create'):
            return cv2.TrackerKCF_create()
        elif hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerKCF_create'):
            return cv2.legacy.TrackerKCF_create()
        else:
            raise AttributeError("Le tracker KCF n'est pas disponible. Assurez-vous d'avoir installé opencv-contrib-python.")
            
    elif tracker_type == "CSRT":
        if hasattr(cv2, 'TrackerCSRT_create'):
            return cv2.TrackerCSRT_create()
        elif hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
            return cv2.legacy.TrackerCSRT_create()
        else:
            raise AttributeError("Le tracker CSRT n'est pas disponible. Assurez-vous d'avoir installé opencv-contrib-python.")
            
    raise ValueError(f"Tracker inconnu : {tracker_type}")


# ─────────────────────────────────────────────────────────────────────────────
# Étape 1 — Signal Farneback + Tracker + vidéo annotée
# ─────────────────────────────────────────────────────────────────────────────

def run_farneback_tracker(video_path, roi, fps, tracker_type,
                           compute_flow, output_path,
                           frame_w, frame_h, no_display):
    """
    Passe 1 : calcul du signal de mouvement Farneback dans la ROI trackée.
    Retourne signal_raw (list[float]).
    """
    tracker = make_tracker(tracker_type)

    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    frame = cv2.resize(frame, (frame_w, frame_h))
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tracker.init(frame, roi)
    current_roi = roi

    signal_raw = []

    print(f"  Passe 1 : signal Farneback ({tracker_type})...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (frame_w, frame_h))
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ok, bbox = tracker.update(frame)
        if ok:
            current_roi = tuple(int(v) for v in bbox)

        flow = compute_flow(prev_gray, gray)

        rx, ry, rw, rh = current_roi
        rx, ry = max(0, rx), max(0, ry)
        roi_flow = flow[ry:ry+rh, rx:rx+rw]
        mag, _ = cv2.cartToPolar(roi_flow[..., 0], roi_flow[..., 1])
        signal_raw.append(float(np.mean(mag)))

        prev_gray = gray

    cap.release()
    print(f"  ✓ {len(signal_raw)} frames analysées")
    return signal_raw


def render_tracking_video(video_path, roi, fps, tracker_type, compute_flow,
                           signal_smooth, peaks, output_path,
                           frame_w, frame_h, no_display):
    """
    Passe 2 : re-lecture avec signal déjà calculé → rendu annoté complet.
    """
    tracker = make_tracker(tracker_type)
    nb_reps   = len(peaks)
    peaks_set = set(peaks.tolist())

    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    frame = cv2.resize(frame, (frame_w, frame_h))
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tracker.init(frame, roi)
    current_roi = roi

    writer    = make_writer(output_path, fps, frame_w, frame_h)
    frame_idx = 0

    print(f"  Rendu vidéo tracking → {output_path.name}...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (frame_w, frame_h))
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ok, bbox = tracker.update(frame)
        if ok:
            current_roi = tuple(int(v) for v in bbox)

        flow = compute_flow(prev_gray, gray)

        rx, ry, rw, rh = current_roi
        rx, ry = max(0, rx), max(0, ry)

        # Flèches de flux dans la ROI (step=12)
        for fy in range(ry, min(ry + rh, frame_h), 12):
            for fx in range(rx, min(rx + rw, frame_w), 12):
                dx, dy = flow[fy, fx]
                cv2.arrowedLine(frame, (fx, fy),
                                (int(fx + dx * 2), int(fy + dy * 2)),
                                (0, 255, 0), 1, tipLength=0.4)

        color = (0, 255, 0) if ok else (0, 0, 255)
        cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), color, 2)

        if frame_idx in peaks_set:
            put_label(frame, "BEAT", (50, 60), scale=1.5, color=(0, 0, 255), thickness=3)

        # Graphe signal incrusté (coin bas-droit)
        graph = _signal_miniature(signal_smooth, peaks, frame_idx, w=320, h=160)
        gh, gw = graph.shape[:2]
        frame[frame_h - gh:, frame_w - gw:] = graph

        reps_so_far = sum(1 for p in peaks if p <= frame_idx)
        put_label(frame, f"Reps: {reps_so_far}/{nb_reps}",
                  (10, frame_h - 12), scale=0.5)

        writer.write(frame)
        if not no_display:
            cv2.imshow("Tracking — Q pour quitter", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break

        prev_gray = gray
        frame_idx += 1

    cap.release()
    writer.release()
    if not no_display:
        cv2.destroyAllWindows()
    print(f"  ✓ {output_path}")


def _signal_miniature(signal, peaks, idx, w=320, h=160):
    """Génère une petite image matplotlib du signal pour incrustation."""
    fig, ax = plt.subplots(figsize=(3.5, 1.8), dpi=80)
    ax.plot(signal[:idx], color="cyan", linewidth=1.0)
    visible = [p for p in peaks if p < idx]
    if visible:
        ax.scatter(visible, signal[visible], color="red", s=12, zorder=5,
                   label=f"{len(visible)} reps")
        ax.legend(fontsize=6, loc="upper left",
                  facecolor="#222", labelcolor="white", edgecolor="none")
    ax.axvline(idx, color="yellow", linewidth=0.8, alpha=0.7)
    ax.set_xlim(0, len(signal))
    ax.set_ylim(0, max(signal.max() * 1.1, 1e-3))
    ax.set_xlabel("Frame", fontsize=6, color="white")
    ax.set_ylabel("Mouvement", fontsize=6, color="white")
    ax.set_title("Signal ROI", fontsize=7, color="white", pad=3)
    ax.tick_params(axis="both", labelsize=5, colors="white")
    ax.spines[:].set_color("#555")
    ax.set_facecolor("#1a1a1a")
    fig.patch.set_facecolor("#1a1a1a")
    fig.tight_layout(pad=0.4)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
    plt.close(fig)
    return cv2.cvtColor(cv2.resize(buf, (w, h)), cv2.COLOR_RGB2BGR)


# ─────────────────────────────────────────────────────────────────────────────
# Étape 2 — Détection des reps
# ─────────────────────────────────────────────────────────────────────────────

def detect_reps(signal_raw, fps):
    signal = np.array(signal_raw, dtype=np.float32)
    signal_smooth = smooth(signal)
    threshold = np.mean(signal_smooth) + 0.5 * np.std(signal_smooth)
    peaks, _ = find_peaks(signal_smooth, distance=10, height=threshold)

    nb_reps = len(peaks)
    cadence = nb_reps / (len(signal) / fps / 60) if len(signal) > 0 else 0

    inter_peak = np.diff(peaks) / fps if nb_reps > 1 else np.array([])
    avg_rep_duration = float(np.mean(inter_peak)) if len(inter_peak) > 0 else 0.0

    print(f"  Répétitions : {nb_reps} | Cadence : {cadence:.1f} reps/min"
          f" | Durée moy./rep : {avg_rep_duration:.2f}s")
    return signal, signal_smooth, peaks, threshold, nb_reps, cadence, avg_rep_duration, inter_peak


# ─────────────────────────────────────────────────────────────────────────────
# Étape 3 — Lucas-Kanade sparse
# ─────────────────────────────────────────────────────────────────────────────

SHITOMASI = dict(maxCorners=100, qualityLevel=0.01, minDistance=7, blockSize=7)
LK_PARAMS = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))


def detect_points_in_roi(gray, roi):
    x, y, w, h = roi
    mask = np.zeros_like(gray)
    mask[y:y+h, x:x+w] = 255
    return cv2.goodFeaturesToTrack(gray, mask=mask, **SHITOMASI)


def run_lucas_kanade(video_path, roi, fps, redetect_every, frame_w, frame_h, csv_path):
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    frame     = cv2.resize(frame, (frame_w, frame_h))
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_pts  = detect_points_in_roi(prev_gray, roi)
    assert prev_pts is not None, "Aucun point Shi-Tomasi dans la ROI — agrandis-la"
    print(f"  {len(prev_pts)} points Shi-Tomasi détectés")

    signal_lk = []; signal_dx = []; signal_dy = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (frame_w, frame_h))
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, prev_pts, None, **LK_PARAMS
        )
        good_prev = prev_pts[status == 1]
        good_next = next_pts[status == 1]

        if len(good_prev) > 0:
            disp = good_next - good_prev
            signal_lk.append(float(np.mean(np.sqrt(disp[:,0]**2 + disp[:,1]**2))))
            signal_dx.append(float(np.mean(disp[:, 0])))
            signal_dy.append(float(np.mean(disp[:, 1])))
        else:
            signal_lk.append(0.0); signal_dx.append(0.0); signal_dy.append(0.0)

        if frame_idx % redetect_every == 0:
            new_pts = detect_points_in_roi(gray, roi)
            prev_pts = new_pts if new_pts is not None else good_next.reshape(-1, 1, 2)
        else:
            prev_pts = good_next.reshape(-1, 1, 2)

        prev_gray = gray
        frame_idx += 1

    cap.release()

    signal_lk = np.array(signal_lk, dtype=np.float32)
    signal_dx = np.array(signal_dx, dtype=np.float32)
    signal_dy = np.array(signal_dy, dtype=np.float32)

    # Export CSV
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "mag", "dx", "dy"])
        for i in range(len(signal_lk)):
            w.writerow([i, signal_lk[i], signal_dx[i], signal_dy[i]])
    print(f"  ✓ {csv_path.name} sauvegardé ({len(signal_lk)} frames)")
    return signal_lk, signal_dx, signal_dy


# ─────────────────────────────────────────────────────────────────────────────
# Étape 4 — Phases du mouvement (dy → concentrique/excentrique)
# ─────────────────────────────────────────────────────────────────────────────

def analyse_phases(signal_dy, fps):
    dy_smooth = smooth(signal_dy)
    phase = np.where(dy_smooth < 0, 1, -1)
    transitions = np.where(np.diff(phase) != 0)[0]
    phase_durations = np.diff(transitions) / fps if len(transitions) > 1 else np.array([])
    up_dur   = phase_durations[phase[transitions[:-1]] == 1]  if len(phase_durations) else np.array([])
    down_dur = phase_durations[phase[transitions[:-1]] == -1] if len(phase_durations) else np.array([])
    if len(up_dur):
        print(f"  Phase montée  : {up_dur.mean():.2f}s moy.")
    if len(down_dur):
        print(f"  Phase descente: {down_dur.mean():.2f}s moy.")
    return dy_smooth, phase, up_dur, down_dur


# ─────────────────────────────────────────────────────────────────────────────
# Étape 5 — Vidéo side-by-side couleurs HSV
# ─────────────────────────────────────────────────────────────────────────────

def render_colored_video(video_path, roi, fps, compute_flow, output_path,
                          frame_w, frame_h, no_display):
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    frame     = cv2.resize(frame, (frame_w, frame_h))
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    writer = make_writer(output_path, fps, frame_w * 2, frame_h)
    rx, ry, rw, rh = roi

    print(f"  Rendu vidéo couleurs → {output_path.name}...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (frame_w, frame_h))
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow     = compute_flow(prev_gray, gray)
        flow_img = flow_to_hsv_bgr(flow)

        cv2.rectangle(frame,    (rx, ry), (rx+rw, ry+rh), (255,255,255), 1)
        cv2.rectangle(flow_img, (rx, ry), (rx+rw, ry+rh), (255,255,255), 1)
        put_label(frame,    "Original",   (10, 20))
        put_label(flow_img, "Flux (HSV)", (10, 20))

        combined = np.hstack([frame, flow_img])
        writer.write(combined)

        if not no_display:
            cv2.imshow("Original | Flux HSV — Q", combined)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break

        prev_gray = gray

    cap.release()
    writer.release()
    if not no_display:
        cv2.destroyAllWindows()
    print(f"  ✓ {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Étape 6 — Heatmap accumulée
# ─────────────────────────────────────────────────────────────────────────────

def compute_heatmap(video_path, compute_flow, frame_w, frame_h):
    """Accumule la magnitude du flux optique pixel par pixel."""
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    frame     = cv2.resize(frame, (frame_w, frame_h))
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    heatmap  = np.zeros((frame_h, frame_w), dtype=np.float32)
    n_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (frame_w, frame_h))
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = compute_flow(prev_gray, gray)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        heatmap += mag
        prev_gray = gray
        n_frames += 1

    cap.release()
    heatmap /= max(n_frames, 1)
    print(f"  ✓ Heatmap calculée sur {n_frames} frames")
    return heatmap, n_frames


def save_heatmap_figure(heatmap, first_frame, roi, n_frames, output_path):
    heatmap_norm  = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    overlay       = cv2.addWeighted(first_frame, 0.4, heatmap_color, 0.6, 0)
    rx, ry, rw, rh = roi
    cv2.rectangle(overlay, (rx, ry), (rx+rw, ry+rh), (255,255,255), 2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].imshow(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Frame originale"); axes[0].axis("off")
    im = axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Heatmap brute"); axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.03)
    axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Superposition — zones actives"); axes[2].axis("off")
    plt.suptitle(f"Heatmap accumulée sur {n_frames} frames", fontsize=13)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {output_path.name}")
    return heatmap_color


# ─────────────────────────────────────────────────────────────────────────────
# Étape 7 — Background subtraction (MOG2) + trajectoire centroïde
# ─────────────────────────────────────────────────────────────────────────────

def run_bgs(video_path, fps, output_path, frame_w, frame_h, no_display):
    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=100, varThreshold=50, detectShadows=False
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    cap = cv2.VideoCapture(str(video_path))
    writer = make_writer(output_path, fps, frame_w * 2, frame_h)

    traj_x = []; traj_y = []; fg_signal = []

    print(f"  Background subtraction → {output_path.name}...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame   = cv2.resize(frame, (frame_w, frame_h))
        fg_mask = bg_sub.apply(frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_signal.append(int(np.sum(fg_mask > 0)))

        M = cv2.moments(fg_mask)
        if M["m00"] > 0:
            traj_x.append(int(M["m10"] / M["m00"]))
            traj_y.append(int(M["m01"] / M["m00"]))
        elif traj_x:
            traj_x.append(traj_x[-1])
            traj_y.append(traj_y[-1])

        annotated = frame.copy()
        for k in range(1, len(traj_x)):
            alpha = k / len(traj_x)
            c = (0, int(255 * alpha), int(255 * (1 - alpha)))
            cv2.line(annotated,
                     (traj_x[k-1], traj_y[k-1]),
                     (traj_x[k],   traj_y[k]), c, 2)
        if traj_x:
            cv2.circle(annotated, (traj_x[-1], traj_y[-1]), 5, (0,255,255), -1)

        fg_bgr = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
        put_label(fg_bgr,   "Foreground",  (10, 30))
        put_label(annotated,"Trajectoire", (10, 30))
        combined = np.hstack([annotated, fg_bgr])
        writer.write(combined)

        if not no_display:
            cv2.imshow("BGS — Q", combined)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break

    cap.release()
    writer.release()
    if not no_display:
        cv2.destroyAllWindows()

    traj_x = np.array(traj_x, dtype=np.int32)
    traj_y = np.array(traj_y, dtype=np.int32)
    fg_signal = np.array(fg_signal, dtype=np.float32)
    print(f"  ✓ {output_path.name} | {len(traj_x)} positions centroïde")
    return traj_x, traj_y, fg_signal


# ─────────────────────────────────────────────────────────────────────────────
# Étape 8 — Multi-ROI + corrélation croisée
# ─────────────────────────────────────────────────────────────────────────────

def auto_multi_rois(frame_w, frame_h, n):
    """Génère n ROIs réparties horizontalement (fallback automatique)."""
    rw = frame_w // (n + 1)
    rh = frame_h // 3
    ry = frame_h // 3
    rois = []
    for i in range(n):
        rx = int(frame_w * (i + 1) / (n + 1)) - rw // 2
        rois.append((rx, ry, rw, rh))
    return rois


def run_multi_roi(video_path, rois, fps, compute_flow, frame_w, frame_h):
    n = len(rois)
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    frame     = cv2.resize(frame, (frame_w, frame_h))
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    signals = [[] for _ in range(n)]

    print(f"  Multi-ROI ({n} zones)...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (frame_w, frame_h))
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow  = compute_flow(prev_gray, gray)

        for i, (rx, ry, rw, rh) in enumerate(rois):
            roi_flow = flow[ry:ry+rh, rx:rx+rw]
            mag, _   = cv2.cartToPolar(roi_flow[..., 0], roi_flow[..., 1])
            signals[i].append(float(np.mean(mag)))

        prev_gray = gray

    cap.release()

    signals_smooth = [smooth(np.array(s, dtype=np.float32)) for s in signals]

    # Corrélation croisée entre toutes les paires
    corr_results = {}
    for i, j in combinations(range(n), 2):
        a = signals_smooth[i] - signals_smooth[i].mean()
        b = signals_smooth[j] - signals_smooth[j].mean()
        corr = np.correlate(a, b, mode="full")
        denom = np.std(a) * np.std(b) * len(a)
        if denom > 0:
            corr /= denom
        lag  = (np.argmax(corr) - (len(a) - 1)) / fps
        peak = float(corr.max())
        corr_results[(i, j)] = (corr, lag, peak)
        print(f"  ROI {i+1} vs ROI {j+1} — corr max : {peak:.3f} | lag : {lag:+.2f}s")

    return signals_smooth, corr_results


# ─────────────────────────────────────────────────────────────────────────────
# Étape 9 — Dashboard final
# ─────────────────────────────────────────────────────────────────────────────

def save_dashboard(
    signal_smooth, signal_lk_smooth, signal_dx, signal_dy,
    peaks, nb_reps, cadence, avg_rep_duration,
    inter_peak, up_dur, down_dur,
    heatmap, roi,
    traj_x, traj_y, fg_signal,
    signals_smooth_multi, corr_results,
    fps, total_frames, frame_w, frame_h,
    output_path
):
    ROI_COLORS_BGR = [(0, 255, 0), (0, 165, 255), (0, 0, 255)]

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("Dashboard — Analyse du mouvement", fontsize=15, fontweight="bold")
    gs  = GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

    times_main = np.arange(len(signal_smooth)) / fps

    # ── [0,0:2] Farneback + LK superposés ─────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(times_main, signal_smooth, color="steelblue", label="Farneback", alpha=0.8)
    if signal_lk_smooth is not None:
        t_lk = np.arange(len(signal_lk_smooth)) / fps
        ax1.plot(t_lk, signal_lk_smooth, color="tomato", label="Lucas-Kanade", alpha=0.8)
    ax1.scatter(times_main[peaks], signal_smooth[peaks], color="red", s=25, zorder=5)
    ax1.set_title("Signal de mouvement (Farneback vs LK)")
    ax1.set_xlabel("Temps (s)"); ax1.set_ylabel("Magnitude")
    ax1.legend(fontsize=8)

    # ── [0,2:4] Phases concentrique / excentrique ──────────────────────────
    ax2 = fig.add_subplot(gs[0, 2:])
    if signal_dy is not None:
        dy_s = smooth(signal_dy)
        t_dy = np.arange(len(dy_s)) / fps
        ax2.fill_between(t_dy, dy_s, 0, where=(dy_s < 0), alpha=0.4,
                         color="green", label="↑ montée")
        ax2.fill_between(t_dy, dy_s, 0, where=(dy_s > 0), alpha=0.4,
                         color="orange", label="↓ descente")
        ax2.axhline(0, color="gray", linewidth=0.5)
        ax2.legend(fontsize=8)
    ax2.set_title("Phases du mouvement (dy)")
    ax2.set_xlabel("Temps (s)"); ax2.set_ylabel("dy (px/frame)")

    # ── [1,0] Durée par rep ────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    if nb_reps > 1 and len(inter_peak):
        bars = ax3.bar(range(1, nb_reps), inter_peak, color="steelblue", alpha=0.8)
        ax3.axhline(inter_peak.mean(), color="orange", linestyle="--",
                    linewidth=1, label="Moyenne")
        fatigue_thr = inter_peak.mean() * 1.2
        for bar, val in zip(bars, inter_peak):
            if val > fatigue_thr:
                bar.set_color("tomato")
        ax3.legend(fontsize=7)
    ax3.set_title("Durée/rep (rouge = fatigue)")
    ax3.set_xlabel("Rep n°"); ax3.set_ylabel("Durée (s)")

    # ── [1,1] Heatmap ──────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    if heatmap is not None:
        ax4.imshow(heatmap, cmap="jet", aspect="auto")
        rx, ry, rw, rh = roi
        ax4.add_patch(plt.Rectangle((rx, ry), rw, rh,
                                     edgecolor="white", facecolor="none", lw=1.5))
    ax4.set_title("Heatmap accumulée"); ax4.axis("off")

    # ── [1,2:4] Trajectoire 2D centroïde ──────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2:])
    if traj_x is not None and len(traj_x):
        sc = ax5.scatter(traj_x, traj_y,
                         c=np.arange(len(traj_x)), cmap="plasma", s=2)
        ax5.invert_yaxis()
        ax5.set_xlim(0, frame_w); ax5.set_ylim(frame_h, 0)
        plt.colorbar(sc, ax=ax5, label="Frame", fraction=0.025)
    ax5.set_title("Trajectoire 2D (centre de masse)")
    ax5.set_xlabel("x (px)"); ax5.set_ylabel("y (px)")

    # ── [2,0] Métriques texte ──────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.axis("off")
    lines = [
        f"Durée totale     : {times_main[-1]:.1f} s",
        f"Répétitions      : {nb_reps}",
        f"Cadence          : {cadence:.1f} reps/min",
        f"Durée moy./rep   : {avg_rep_duration:.2f} s",
        f"Montée moy.      : {up_dur.mean():.2f} s"   if len(up_dur)   else "Montée moy.      : —",
        f"Descente moy.    : {down_dur.mean():.2f} s" if len(down_dur) else "Descente moy.    : —",
        f"FPS              : {fps:.0f}",
        f"Frames analysées : {total_frames}",
    ]
    ax6.text(0.05, 0.95, "\n".join(lines),
             transform=ax6.transAxes, fontsize=9, verticalalignment="top",
             fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#1a1a2e", alpha=0.8))
    ax6.set_title("Métriques")

    # ── [2,1] Pixels foreground ────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 1])
    if fg_signal is not None and len(fg_signal):
        fg_s = smooth(fg_signal)
        t_fg = np.arange(len(fg_s)) / fps
        ax7.plot(t_fg, fg_s / max(fg_s.max(), 1e-6), color="mediumpurple")
    ax7.set_title("Intensité globale (BGS)")
    ax7.set_xlabel("Temps (s)"); ax7.set_ylabel("Foreground normalisé")

    # ── [2,2:4] Asymétrie multi-ROI ───────────────────────────────────────
    ax8 = fig.add_subplot(gs[2, 2:])
    if signals_smooth_multi is not None and len(signals_smooth_multi) >= 2:
        t_roi = np.arange(len(signals_smooth_multi[0])) / fps
        diff  = signals_smooth_multi[0] - signals_smooth_multi[1]
        ax8.fill_between(t_roi, diff, 0, where=(diff > 0), alpha=0.4,
                         color="green", label="ROI 1 dominant")
        ax8.fill_between(t_roi, diff, 0, where=(diff < 0), alpha=0.4,
                         color="orange", label="ROI 2 dominant")
        ax8.axhline(0, color="gray", linewidth=0.5)
        ax8.legend(fontsize=8)
        ax8.set_title("Asymétrie ROI 1 vs ROI 2")
        ax8.set_xlabel("Temps (s)"); ax8.set_ylabel("Différence")
    else:
        ax8.text(0.5, 0.5, "Multi-ROI non exécuté\n(--no-multiroi)",
                 ha="center", va="center", transform=ax8.transAxes,
                 color="gray", fontsize=10)
        ax8.axis("off")

    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {output_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    VIDEO_PATH = Path(args.video)
    OUTPUT_DIR = Path(args.output_dir)
    FRAME_W    = args.width
    FRAME_H    = args.height

    assert VIDEO_PATH.exists(), f"Vidéo introuvable : {VIDEO_PATH}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Vidéo    : {VIDEO_PATH}")
    print(f"  Sortie   : {OUTPUT_DIR}")
    print(f"  Tracker  : {args.tracker}")
    print(f"{'='*60}\n")

    # ── Farneback (CPU unquement) ───────────────────────────────
    compute_flow = build_farneback()

    # ── Lecture vidéo ──────────────────────────────────────────────────────
    print("[1/8] Lecture vidéo...")
    _, fps, total_frames = open_video(VIDEO_PATH, FRAME_W, FRAME_H)
    first_frame = read_first_frame(VIDEO_PATH, FRAME_W, FRAME_H)

    # ── ROI ────────────────────────────────────────────────────────────────
    if args.roi:
        roi = tuple(args.roi)
        print(f"  ROI manuelle : {roi}")
    else:
        roi = auto_roi(FRAME_W, FRAME_H)
        print(f"  ROI automatique (centre) : {roi}")

    # ── Farneback + tracker → signal brut ─────────────────────────────────
    print("\n[2/8] Signal Farneback + tracker...")
    signal_raw = run_farneback_tracker(
        VIDEO_PATH, roi, fps, args.tracker, compute_flow,
        OUTPUT_DIR / "_tmp_pass1.mp4",
        FRAME_W, FRAME_H, no_display=True   # silencieux pour la passe 1
    )

    # Export CSV frames
    csv_frames = OUTPUT_DIR / "frames_output.csv"
    import pandas as pd
    pd.DataFrame({"frame": list(range(len(signal_raw)))}).to_csv(csv_frames, index=False)
    print(f"  ✓ {csv_frames.name}")

    # ── Détection des reps ────────────────────────────────────────────────
    print("\n[3/8] Détection des répétitions...")
    (signal, signal_smooth, peaks, threshold,
     nb_reps, cadence, avg_rep_duration, inter_peak) = detect_reps(signal_raw, fps)

    # ── Rendu vidéo annoté complet ────────────────────────────────────────
    print("\n[4/8] Vidéo annotée tracking...")
    render_tracking_video(
        VIDEO_PATH, roi, fps, args.tracker, compute_flow,
        signal_smooth, peaks,
        OUTPUT_DIR / "resultat_tracking.mp4",
        FRAME_W, FRAME_H, no_display=args.no_display
    )

    # ── Vidéo side-by-side HSV ────────────────────────────────────────────
    print("\n[4b] Vidéo couleurs HSV side-by-side...")
    render_colored_video(
        VIDEO_PATH, roi, fps, compute_flow,
        OUTPUT_DIR / "resultat_couleurs.mp4",
        FRAME_W, FRAME_H, no_display=args.no_display
    )

    # ── Lucas-Kanade ──────────────────────────────────────────────────────
    signal_lk = signal_lk_smooth = signal_dx = signal_dy = None
    up_dur = down_dur = np.array([])
    if not args.no_lk:
        print("\n[5/8] Lucas-Kanade sparse...")
        signal_lk, signal_dx, signal_dy = run_lucas_kanade(
            VIDEO_PATH, roi, fps, args.redetect, FRAME_W, FRAME_H,
            OUTPUT_DIR / "optical_flow_lk.csv"
        )
        signal_lk_smooth = smooth(signal_lk)
        print("\n  Analyse des phases (dy)...")
        _, _, up_dur, down_dur = analyse_phases(signal_dy, fps)
    else:
        print("\n[5/8] Lucas-Kanade désactivé.")

    # ── Heatmap ───────────────────────────────────────────────────────────
    heatmap = None
    if not args.no_heatmap:
        print("\n[6/8] Heatmap accumulée...")
        heatmap, n_frames_hm = compute_heatmap(VIDEO_PATH, compute_flow, FRAME_W, FRAME_H)
        save_heatmap_figure(heatmap, first_frame, roi, n_frames_hm,
                            OUTPUT_DIR / "heatmap_mouvement.png")
    else:
        print("\n[6/8] Heatmap désactivée.")

    # ── Background subtraction ────────────────────────────────────────────
    traj_x = traj_y = fg_signal = None
    if not args.no_bgs:
        print("\n[7/8] Background subtraction (MOG2)...")
        traj_x, traj_y, fg_signal = run_bgs(
            VIDEO_PATH, fps,
            OUTPUT_DIR / "resultat_bgs.mp4",
            FRAME_W, FRAME_H, no_display=args.no_display
        )
    else:
        print("\n[7/8] Background subtraction désactivé.")

    # ── Multi-ROI ─────────────────────────────────────────────────────────
    signals_smooth_multi = None
    corr_results         = {}
    if not args.no_multiroi and args.n_rois >= 2:
        print(f"\n[7b] Multi-ROI ({args.n_rois} zones automatiques)...")
        multi_rois = auto_multi_rois(FRAME_W, FRAME_H, args.n_rois)
        signals_smooth_multi, corr_results = run_multi_roi(
            VIDEO_PATH, multi_rois, fps, compute_flow, FRAME_W, FRAME_H
        )
    else:
        print("\n[7b] Multi-ROI désactivé.")

    # ── Dashboard ─────────────────────────────────────────────────────────
    if not args.no_dashboard:
        print("\n[8/8] Dashboard final...")
        save_dashboard(
            signal_smooth, signal_lk_smooth, signal_dx, signal_dy,
            peaks, nb_reps, cadence, avg_rep_duration,
            inter_peak, up_dur, down_dur,
            heatmap, roi,
            traj_x, traj_y, fg_signal,
            signals_smooth_multi, corr_results,
            fps, total_frames, FRAME_W, FRAME_H,
            OUTPUT_DIR / "dashboard_final.png"
        )
    else:
        print("\n[8/8] Dashboard désactivé.")

    # ── Nettoyage fichier temp ─────────────────────────────────────────────
    tmp = OUTPUT_DIR / "_tmp_pass1.mp4"
    if tmp.exists():
        tmp.unlink()

    # ── Résumé ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  TERMINÉ — fichiers générés :")
    for f in sorted(OUTPUT_DIR.iterdir()):
        size_mb = f.stat().st_size / 1e6
        print(f"    {f.name:<40} {size_mb:.1f} MB")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()