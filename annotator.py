#!/usr/bin/env python3
"""
Annotateur de verite terrain — script local
============================================
Usage:
    python annotator.py video.mp4
    python annotator.py video.mp4 --stride 2
    python annotator.py video.mp4 --output mes_pics.csv --jump 10

Dependances:
    pip install opencv-python numpy pandas

Controles dans la fenetre:
    FLECHE GAUCHE / A   : reculer d'1 frame
    FLECHE DROITE / D   : avancer d'1 frame
    , (virgule)         : reculer de N frames (--jump)
    . (point)           : avancer de N frames (--jump)
    ESPACE              : marquer / demarquer la frame courante
    U                   : annuler le dernier marquage
    C                   : effacer tous les marquages
    Q ou ESC            : quitter et exporter le CSV
"""

import sys
import os
import argparse
import cv2
import numpy as np
import pandas as pd


# ── Arguments ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Annotateur de pics de mouvement")
parser.add_argument("video",                          help="Chemin vers la video")
parser.add_argument("--stride",     type=int, default=1,                  help="Charger 1 frame sur N (defaut: 1)")
parser.add_argument("--output",     default="ground_truth.csv",           help="Fichier CSV de sortie")
parser.add_argument("--max-frames", type=int, default=5000,               help="Plafond de frames en cache (defaut: 5000)")
parser.add_argument("--jump",       type=int, default=5,                  help="Taille du saut avec , et . (defaut: 5)")
parser.add_argument("--width",      type=int, default=900,                help="Largeur max d'affichage en px (defaut: 900)")
args = parser.parse_args()

VIDEO_PATH = args.video
STRIDE     = args.stride
OUT_PATH   = args.output
MAX_FRAMES = args.max_frames
JUMP       = args.jump
MAX_W      = args.width


# ── Verification du fichier ───────────────────────────────────────────────────
if not os.path.exists(VIDEO_PATH):
    print(f"Erreur : fichier introuvable : {VIDEO_PATH}")
    sys.exit(1)


# ── Infos video ───────────────────────────────────────────────────────────────
cap      = cv2.VideoCapture(VIDEO_PATH)
FPS      = cap.get(cv2.CAP_PROP_FPS) or 30.0
N_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()
DURATION = N_FRAMES / FPS

print(f"\nVideo : {os.path.basename(VIDEO_PATH)}")
print(f"  {N_FRAMES} frames  |  {FPS:.1f} fps  |  {DURATION:.1f}s  |  {W}x{H}px")
if DURATION < 10:
    print(f"  Attention : {DURATION:.1f}s est tres court — viser au moins 5 evenements")
elif DURATION > 300:
    print(f"  Attention : {DURATION:.1f}s est long — envisager --stride 2 ou 3")


# ── Chargement des frames en cache ────────────────────────────────────────────
print(f"\nChargement (stride={STRIDE}, max={MAX_FRAMES} frames)...")
cap    = cv2.VideoCapture(VIDEO_PATH)
FRAMES = []
fi     = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if fi % STRIDE == 0:
        h, w   = frame.shape[:2]
        disp_w = min(MAX_W, w)
        disp_h = int(h * disp_w / w)
        frame  = cv2.resize(frame, (disp_w, disp_h))
        FRAMES.append(frame)
        if len(FRAMES) >= MAX_FRAMES:
            print(f"  Plafond atteint : {MAX_FRAMES} frames")
            break
    fi += 1
cap.release()

N       = len(FRAMES)
EFF_FPS = FPS / STRIDE
print(f"  {N} frames en cache  (fps effectif : {EFF_FPS:.1f})")
print(f"  RAM ~ {N * FRAMES[0].nbytes / 1e6:.0f} MB")


# ── Etat ──────────────────────────────────────────────────────────────────────
marks   = set()   # ensemble des indices de frames cachees marquees
current = 0       # index courant dans FRAMES


# ── Rendu d'une frame avec overlays ──────────────────────────────────────────
def draw_frame(idx):
    frame     = FRAMES[idx].copy()
    fh, fw    = frame.shape[:2]
    real_frame = idx * STRIDE
    t          = real_frame / FPS
    is_marked  = idx in marks

    # Bande info en haut
    cv2.rectangle(frame, (0, 0), (fw, 38), (23, 17, 15), -1)
    mark_color = (113, 204, 46) if is_marked else (241, 240, 236)  # BGR
    mark_str   = "  [MARQUE]" if is_marked else ""
    info       = f"Frame {real_frame}   t={t:.3f}s   {len(marks)} marquage(s){mark_str}"
    cv2.putText(frame, info, (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, mark_color, 1, cv2.LINE_AA)

    # Bande controles en bas
    cv2.rectangle(frame, (0, fh - 42), (fw, fh), (23, 17, 15), -1)
    ctrl = "ESPACE:marquer  U:undo  C:effacer  A/D ou fleches:+/-1  , . :saut  Q/ESC:quitter"
    cv2.putText(frame, ctrl, (8, fh - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (136, 136, 136), 1, cv2.LINE_AA)

    # Timeline
    tl_y = fh - 30
    tl_h = 10
    cv2.rectangle(frame, (0, tl_y), (fw, tl_y + tl_h), (64, 45, 42), -1)

    # Marquages sur la timeline
    for m in sorted(marks):
        mx = int(m / max(N - 1, 1) * (fw - 1))
        cv2.line(frame, (mx, tl_y), (mx, tl_y + tl_h), (113, 204, 46), 2)

    # Curseur position courante
    cx = int(idx / max(N - 1, 1) * (fw - 1))
    cv2.line(frame, (cx, tl_y - 4), (cx, tl_y + tl_h + 4), (60, 76, 231), 2)

    return frame


# ── Boucle principale ─────────────────────────────────────────────────────────
WINDOW = "Annotateur — " + os.path.basename(VIDEO_PATH)
cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
fh, fw = FRAMES[0].shape[:2]
cv2.resizeWindow(WINDOW, fw, fh)

print("\nControles :")
print("  ESPACE             : marquer / demarquer la frame courante")
print("  U                  : annuler le dernier marquage")
print("  C                  : effacer tous les marquages")
print("  FLECHE G/D  ou A/D : +/- 1 frame")
print(f"  ,  et  .           : +/- {JUMP} frames")
print("  Q ou ESC           : quitter et exporter\n")

while True:
    cv2.imshow(WINDOW, draw_frame(current))

    # waitKeyEx pour capturer les touches speciales (fleches)
    key = cv2.waitKeyEx(0)

    # Fleches (codes cross-platform)
    if key in (65361, 2424832, ord('a'), ord('A')):   # gauche
        current = max(0, current - 1)

    elif key in (65363, 2555904, ord('d'), ord('D')): # droite
        current = min(N - 1, current + 1)

    elif key == ord(','):                              # saut arriere
        current = max(0, current - JUMP)

    elif key == ord('.'):                              # saut avant
        current = min(N - 1, current + JUMP)

    elif key == ord(' '):                              # marquer / demarquer
        if current in marks:
            marks.discard(current)
            print(f"  Demarque : frame {current * STRIDE}  (total={len(marks)})")
        else:
            marks.add(current)
            print(f"  Marque   : frame {current * STRIDE}  t={current * STRIDE / FPS:.3f}s  (total={len(marks)})")

    elif key in (ord('u'), ord('U')):                  # undo dernier marquage
        if marks:
            last = max(marks)
            marks.discard(last)
            print(f"  Undo : frame {last * STRIDE} supprimee  (total={len(marks)})")
        else:
            print("  Rien a annuler")

    elif key in (ord('c'), ord('C')):                  # effacer tout
        marks.clear()
        print("  Tous les marquages effaces")

    elif key in (ord('q'), ord('Q'), 27):              # quitter
        break

cv2.destroyAllWindows()


# ── Export CSV ────────────────────────────────────────────────────────────────
if not marks:
    print("\nAucun marquage — fichier CSV non cree")
    sys.exit(0)

real_peaks = sorted([m * STRIDE for m in marks])
times      = [round(p / FPS, 3) for p in real_peaks]
gaps       = [round(times[i+1] - times[i], 3) for i in range(len(times) - 1)]

gt_df = pd.DataFrame({
    "event_id":        range(1, len(real_peaks) + 1),
    "frame_peak":      real_peaks,
    "time_sec":        times,
    "gap_to_next_sec": gaps + [None],
})

gt_df.to_csv(OUT_PATH, index=False)

print(f"\nExporte {len(real_peaks)} evenements -> {OUT_PATH}")
print()
print(gt_df.to_string(index=False))
print()

if gaps:
    print(f"Ecart moyen : {np.mean(gaps):.3f}s")
    print(f"Min / Max   : {min(gaps):.3f}s  /  {max(gaps):.3f}s")
    if min(gaps) < 0.2:
        print("Attention : certains ecarts < 0.2s — double marquage possible")
    if max(gaps) > 5.0:
        print("Attention : certains ecarts > 5s — evenement manque possible")

print(f"\nPret pour le benchmark : GT_PATH = '{OUT_PATH}'")
