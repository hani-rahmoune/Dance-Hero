#!/usr/bin/env python3
"""
analyse_video.py — Pipeline d'analyse vidéo complet
Basé sur test_stride.ipynb

Usage:
    python analyse_video.py <video_path> [options]

Options:
    --output-dir DIR      Dossier de sortie (défaut: outputs/)
    --max-frames N        Nombre max de frames à traiter (défaut: 300, 0=tout)
    --target-diff FLOAT   Seuil cible pour le stride adaptatif (défaut: 3.0)
    --raft-dir DIR        Chemin vers le repo RAFT (défaut: RAFT/)
    --raft-weights PATH   Chemin vers les poids RAFT (défaut: RAFT/models/raft-sintel.pth)
    --no-raft             Désactiver RAFT (utilise Farneback uniquement)
    --no-vit              Désactiver ViT
    --no-midas            Désactiver MiDaS
    --no-deepsort         Désactiver DeepSORT
    --no-tracking         Désactiver tout le tracking (ViT + DeepSORT)
    --fps-out INT         FPS de la vidéo de sortie (défaut: 30)
    --width INT           Largeur de travail (défaut: 640)
    --height INT          Hauteur de travail (défaut: 360)
    --device STR          Device PyTorch : cuda / cuda:0 / cuda:1 / cpu (défaut: cuda si dispo)
    --amp                 Activer mixed precision (float16) sur GPU — +30-50% vitesse
    --vit-batch INT       Taille de batch pour l'inférence ViT (défaut: 8)
    --midas-batch INT     Taille de batch pour l'inférence MiDaS (défaut: 4)

Sorties dans --output-dir :
    raft_side_by_side.mp4       Vidéo côte à côte : original | RAFT HSV
    farneback_side_by_side.mp4  Vidéo côte à côte : original | Farneback HSV
    resultat_3d.mp4             Overlay : tracker ViT + profondeur MiDaS
    resultat_deepsort.mp4       Tracking multi-objets DeepSORT + YOLO
    trajectoire_3d.html         Trajectoire 3D interactive (Plotly)
    trajectoire_raft_midas.html Trajectoire 3D RAFT+MiDaS (Plotly)
    fusion_features.png         Courbes fusionnées par objet
"""

import sys, os, argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")   # pas de display — on sauvegarde les figures
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

import torch

# ─────────────────────────────────────────────────────────────────────────────
# Parsing des arguments
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Analyse vidéo complète (RAFT + ViT + MiDaS + DeepSORT)")
    p.add_argument("video", help="Chemin vers la vidéo source")
    p.add_argument("--output-dir",    default="outputs", help="Dossier de sortie")
    p.add_argument("--max-frames",    type=int,   default=300,  help="Limite de frames (0 = tout)")
    p.add_argument("--target-diff",   type=float, default=3.0,  help="Cible diff L1 pour stride adaptatif")
    p.add_argument("--raft-dir",      default="RAFT",                    help="Répertoire RAFT")
    p.add_argument("--raft-weights",  default="RAFT/models/raft-sintel.pth", help="Poids RAFT")
    p.add_argument("--no-raft",       action="store_true", help="Désactiver RAFT")
    p.add_argument("--no-vit",        action="store_true", help="Désactiver ViT")
    p.add_argument("--no-midas",      action="store_true", help="Désactiver MiDaS")
    p.add_argument("--no-deepsort",   action="store_true", help="Désactiver DeepSORT")
    p.add_argument("--no-tracking",   action="store_true", help="Désactiver ViT + DeepSORT")
    p.add_argument("--fps-out",      type=int,   default=30,   help="FPS vidéo de sortie")
    p.add_argument("--width",        type=int,   default=640,  help="Largeur de travail")
    p.add_argument("--height",       type=int,   default=360,  help="Hauteur de travail")
    p.add_argument("--device",       default=None,
                   help="Device PyTorch : cuda / cuda:0 / cuda:1 / cpu (défaut: cuda si dispo)")
    p.add_argument("--amp",          action="store_true",
                   help="Mixed precision float16 sur GPU (+30-50%% vitesse, Ampere+)")
    p.add_argument("--vit-batch",    type=int, default=8,
                   help="Batch size inférence ViT (défaut: 8)")
    p.add_argument("--midas-batch",  type=int, default=4,
                   help="Batch size inférence MiDaS (défaut: 4)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires
# ─────────────────────────────────────────────────────────────────────────────

def flow_to_hsv_bgr(flow: np.ndarray) -> np.ndarray:
    """Convertit un champ de flux (H,W,2) en image HSV BGR uint8."""
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def id_to_color(track_id):
    np.random.seed(abs(hash(str(track_id))) % (2**31))
    return tuple(int(x) for x in np.random.randint(80, 255, 3))

def make_writer(path, fps, w, h):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    assert writer.isOpened(), f"Impossible de créer {path}"
    return writer

def put_label(frame, text, pos, scale=0.7, color=(255,255,255), thickness=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


# ─────────────────────────────────────────────────────────────────────────────
# Étape 1 — Chargement vidéo
# ─────────────────────────────────────────────────────────────────────────────

def load_video(video_path, max_frames, frame_w, frame_h):
    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), f"Impossible d'ouvrir la vidéo : {video_path}"

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_native     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_native     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration     = total_frames / fps

    print(f"  Résolution native  : {w_native}x{h_native}")
    print(f"  FPS : {fps:.1f} | Frames : {total_frames} | Durée : {duration:.1f}s")

    frames_bgr = []
    frames_rgb = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and len(frames_bgr) >= max_frames:
            break
        frame = cv2.resize(frame, (frame_w, frame_h))
        frames_bgr.append(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        frames_rgb.append(rgb)

    cap.release()
    N = len(frames_bgr)
    print(f"  ✓ {N} frames extraites ({N/fps:.1f}s)")
    return frames_bgr, frames_rgb, fps, N


# ─────────────────────────────────────────────────────────────────────────────
# Étape 2 — Stride adaptatif
# ─────────────────────────────────────────────────────────────────────────────

def compute_adaptive_stride(frames_bgr, fps, N, target_diff):
    print("  Analyse du changement inter-frames...")
    diffs = []
    for i in range(N - 1):
        g1 = cv2.cvtColor(frames_bgr[i],   cv2.COLOR_BGR2GRAY).astype(np.float32)
        g2 = cv2.cvtColor(frames_bgr[i+1], cv2.COLOR_BGR2GRAY).astype(np.float32)
        diffs.append(float(np.mean(np.abs(g2 - g1))))
    diffs = np.array(diffs)

    diff_per_frame = float(np.median(diffs))
    if diff_per_frame < 1e-6:
        stride = 1
    else:
        stride = max(1, min(8, round(target_diff / diff_per_frame)))

    print(f"  diff médiane / frame : {diff_per_frame:.3f}")
    print(f"  TARGET_DIFF         : {target_diff}")
    print(f"  → AUTO_STRIDE       : {stride}")
    return stride, diffs


# ─────────────────────────────────────────────────────────────────────────────
# Étape 3 — RAFT
# ─────────────────────────────────────────────────────────────────────────────

def load_raft(raft_dir, weights_path, device):
    import importlib, argparse as ap
    sys.path.insert(0, str(Path(raft_dir) / "core"))
    raft_mod = importlib.import_module("raft")
    RAFT = raft_mod.RAFT
    utils_mod = importlib.import_module("utils.utils")
    InputPadder = utils_mod.InputPadder

    args = ap.Namespace(small=False, mixed_precision=False, alternate_corr=False)
    model = RAFT(args)
    weights = torch.load(weights_path, map_location=device, weights_only=True)
    weights = {k.replace("module.", ""): v for k, v in weights.items()}
    model.load_state_dict(weights)
    model = model.to(device).eval()
    print(f"  ✓ RAFT chargé sur {device}")
    return model, InputPadder

def compute_raft_flows(frames_bgr, stride, N, device, raft_model, InputPadder, frame_w, frame_h, use_amp=False):
    def to_tensor(bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2,0,1).float()[None]
        if device.type == "cuda":
            t = t.pin_memory()
        return t.to(device, non_blocking=True)

    flows  = []
    signal = []
    autocast_ctx = torch.cuda.amp.autocast() if use_amp else torch.amp.autocast("cpu", enabled=False)
    print(f"  Calcul RAFT sur {N // stride} paires (stride={stride})...")
    for i in range(0, N - stride, stride):
        t1, t2 = to_tensor(frames_bgr[i]), to_tensor(frames_bgr[i+stride])
        padder = InputPadder(t1.shape)
        t1, t2 = padder.pad(t1, t2)
        with torch.no_grad(), autocast_ctx:
            _, flow = raft_model(t1, t2, iters=20, test_mode=True)
        flow = padder.unpad(flow)
        flow_np = flow[0].float().permute(1,2,0).cpu().numpy()
        flows.append(flow_np)
        mag = np.sqrt(flow_np[...,0]**2 + flow_np[...,1]**2)
        signal.append(float(np.mean(mag)))
        if (len(flows)) % 30 == 0:
            print(f"    {len(flows)}/{N // stride}")
            if device.type == "cuda":
                torch.cuda.empty_cache()
    print(f"  ✓ {len(flows)} flux RAFT calculés")
    return flows, np.array(signal, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Étape 4 — Farneback
# ─────────────────────────────────────────────────────────────────────────────

def compute_farneback_flows(frames_bgr, stride, N):
    flows  = []
    signal = []
    print(f"  Calcul Farneback sur {N // stride} paires...")
    for i in range(0, N - stride, stride):
        g1 = cv2.cvtColor(frames_bgr[i],       cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(frames_bgr[i+stride], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            g1, g2, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        flows.append(flow)
        mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
        signal.append(float(np.mean(mag)))
    print(f"  ✓ {len(flows)} flux Farneback calculés")
    return flows, np.array(signal, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Vidéo side-by-side générique
# ─────────────────────────────────────────────────────────────────────────────

def make_side_by_side_video(frames_bgr, flows, stride, N, output_path, fps_out,
                             frame_w, frame_h, label_right="Optical Flow"):
    """Génère une vidéo : original (gauche) | visualisation flux HSV (droite)."""
    out_w = frame_w * 2
    writer = make_writer(output_path, fps_out, out_w, frame_h)
    print(f"  Génération {output_path.name} ({len(flows)} frames)...")

    for i, flow in enumerate(flows):
        frame_idx = i * stride
        left  = frames_bgr[frame_idx].copy()
        right = flow_to_hsv_bgr(flow)

        frame_out = np.concatenate([left, right], axis=1)

        put_label(frame_out, "Original",           (10, 30))
        put_label(frame_out, label_right,           (frame_w + 10, 30))
        put_label(frame_out, f"Frame {frame_idx}",  (10, frame_h - 10),
                  scale=0.5, color=(200,200,200), thickness=1)

        writer.write(frame_out)
        if (i+1) % 30 == 0:
            print(f"    {i+1}/{len(flows)}")

    writer.release()
    print(f"  ✓ {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Étape 5 — ViT features
# ─────────────────────────────────────────────────────────────────────────────

def load_vit(device):
    from transformers import ViTModel, ViTImageProcessor
    MODEL_NAME = "google/vit-base-patch16-224"
    print("  Chargement ViT...")
    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    model     = ViTModel.from_pretrained(MODEL_NAME).to(device).eval()
    print(f"  ✓ ViT chargé sur {device}")
    return model, processor

def extract_all_vit_features(frames_bgr, N, device, vit_model, processor, batch_size=8, use_amp=False):
    cls_tokens   = []
    patch_tokens = []
    autocast_ctx = torch.cuda.amp.autocast() if use_amp else torch.amp.autocast("cpu", enabled=False)
    print(f"  Extraction features ViT sur {N} frames (batch={batch_size})...")

    for batch_start in range(0, N, batch_size):
        batch_frames = frames_bgr[batch_start : batch_start + batch_size]
        batch_rgb    = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in batch_frames]

        inputs = processor(images=batch_rgb, return_tensors="pt", padding=True)
        if device.type == "cuda":
            inputs = {k: v.pin_memory() for k, v in inputs.items()}
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

        with torch.no_grad(), autocast_ctx:
            outputs = vit_model(**inputs)

        # last_hidden_state : (B, 197, 768)
        hs = outputs.last_hidden_state.float().cpu()
        for b in range(hs.shape[0]):
            cls_tokens.append(hs[b, 0].numpy())
            patch_tokens.append(hs[b, 1:].numpy())

        done = min(batch_start + batch_size, N)
        if done % 30 < batch_size:
            print(f"    {done}/{N}")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    cls_tokens   = np.stack(cls_tokens)    # (N, 768)
    patch_tokens = np.stack(patch_tokens)  # (N, 196, 768)
    sim_signal   = np.array([cosine_sim(cls_tokens[i], cls_tokens[i+1]) for i in range(N-1)], dtype=np.float32)
    change_signal = 1 - sim_signal
    print(f"  ✓ ViT features extraites")
    return cls_tokens, patch_tokens, change_signal


# ─────────────────────────────────────────────────────────────────────────────
# Étape 6 — MiDaS profondeur
# ─────────────────────────────────────────────────────────────────────────────

def load_midas(device):
    print("  Chargement MiDaS DPT_Hybrid...")
    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", pretrained=True, trust_repo=True).to(device).eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    print(f"  ✓ MiDaS chargé sur {device}")
    return midas, transforms.dpt_transform

def compute_depth_maps(frames_bgr, N, device, midas, transform, frame_w, frame_h, batch_size=4, use_amp=False):
    print(f"  Estimation profondeur sur {N} frames (batch={batch_size})...")
    depth_maps   = []
    autocast_ctx = torch.cuda.amp.autocast() if use_amp else torch.amp.autocast("cpu", enabled=False)

    for batch_start in range(0, N, batch_size):
        batch_frames = frames_bgr[batch_start : batch_start + batch_size]
        tensors = []
        for frame in batch_frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t   = transform(rgb)           # (1, 3, H', W')  ou  (3, H', W')
            if t.dim() == 3:
                t = t.unsqueeze(0)
            tensors.append(t)
        # Batch : toutes les frames ont la même taille après transform
        batch_tensor = torch.cat(tensors, dim=0)   # (B, 3, H', W')
        if device.type == "cuda":
            batch_tensor = batch_tensor.pin_memory()
        batch_tensor = batch_tensor.to(device, non_blocking=True)

        with torch.no_grad(), autocast_ctx:
            pred = midas(batch_tensor)     # (B, H', W')
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1).float(),
                size=(frame_h, frame_w),
                mode="bicubic", align_corners=False
            ).squeeze(1)                   # (B, H, W)

        pred_np = pred.cpu().float().numpy()
        for b in range(pred_np.shape[0]):
            d = pred_np[b]
            d = (d - d.min()) / (d.max() - d.min() + 1e-8)
            depth_maps.append(d)

        done = min(batch_start + batch_size, N)
        if done % 30 < batch_size:
            print(f"    {done}/{N}")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    depth_maps = np.stack(depth_maps)
    print(f"  ✓ {len(depth_maps)} cartes de profondeur")
    return depth_maps


# ─────────────────────────────────────────────────────────────────────────────
# Étape 7 — Tracking ViT dans une ROI (ROI automatique = centre)
# ─────────────────────────────────────────────────────────────────────────────

def auto_roi(frame_w, frame_h, ratio=0.25):
    """ROI centrale de taille ratio*W x ratio*H."""
    rw = int(frame_w * ratio)
    rh = int(frame_h * ratio)
    rx = (frame_w - rw) // 2
    ry = (frame_h - rh) // 2
    return rx, ry, rw, rh

def vit_track(patch_tokens, N, rx, ry, rw, rh, frame_w, frame_h):
    N_PATCHES_SIDE = 14
    PATCH_W = frame_w / N_PATCHES_SIDE
    PATCH_H = frame_h / N_PATCHES_SIDE

    def patches_in_roi(rx, ry, rw, rh):
        indices = []
        for row in range(N_PATCHES_SIDE):
            for col in range(N_PATCHES_SIDE):
                px = col * PATCH_W; py = row * PATCH_H
                if (px < rx+rw and px+PATCH_W > rx and
                    py < ry+rh and py+PATCH_H > ry):
                    indices.append(row * N_PATCHES_SIDE + col)
        return indices

    ref_indices = patches_in_roi(rx, ry, rw, rh)
    ref_feats   = patch_tokens[0][ref_indices].mean(axis=0)

    roi_w_p = max(1, round(rw / PATCH_W))
    roi_h_p = max(1, round(rh / PATCH_H))

    def find_best_match(frame_patch_feats):
        best_sim = -1; best_pos = (0, 0)
        for row in range(N_PATCHES_SIDE - roi_h_p + 1):
            for col in range(N_PATCHES_SIDE - roi_w_p + 1):
                idxs = [(row+dr)*N_PATCHES_SIDE + (col+dc)
                        for dr in range(roi_h_p) for dc in range(roi_w_p)]
                feats = frame_patch_feats[idxs].mean(axis=0)
                sim   = cosine_sim(feats, ref_feats)
                if sim > best_sim:
                    best_sim = sim; best_pos = (col, row)
        return best_pos, best_sim

    track_positions = []; similarities = []
    print(f"  Tracking ViT sur {N} frames...")
    for i in range(N):
        (col, row), sim = find_best_match(patch_tokens[i])
        track_positions.append((int(col * PATCH_W), int(row * PATCH_H)))
        similarities.append(sim)
        if (i+1) % 30 == 0:
            print(f"    {i+1}/{N}")
    print(f"  ✓ Tracking ViT terminé | sim moy. : {np.mean(similarities):.4f}")
    return track_positions, similarities, roi_w_p, roi_h_p, PATCH_W, PATCH_H


# ─────────────────────────────────────────────────────────────────────────────
# Étape 8 — Vidéo overlay 3D (ViT + MiDaS)
# ─────────────────────────────────────────────────────────────────────────────

def make_3d_overlay_video(frames_bgr, track_positions, similarities, depth_maps,
                           traj_x, traj_y, traj_z, roi_w_p, roi_h_p,
                           PATCH_W, PATCH_H, N, fps_out, output_path, frame_w, frame_h):
    cmap     = plt.cm.plasma
    TRAIL_LEN = 40
    writer = make_writer(output_path, fps_out, frame_w, frame_h)
    print(f"  Génération {output_path.name}...")

    def depth_to_bgr(d):
        r, g, b, _ = cmap(float(d))
        return (int(b*255), int(g*255), int(r*255))

    for i in range(N):
        frame = frames_bgr[i].copy()
        d     = float(traj_z[i]) if i < len(traj_z) else 0.5
        color = depth_to_bgr(d)

        start = max(0, i - TRAIL_LEN)
        for k in range(start+1, i+1):
            if k < len(traj_x):
                alpha = (k - start) / TRAIL_LEN
                c     = depth_to_bgr(float(traj_z[k]) if k < len(traj_z) else d)
                cv2.circle(frame, (int(traj_x[k]), int(traj_y[k])), int(2+alpha*3), c, -1)

        x, y   = track_positions[i]
        w_px   = int(roi_w_p * PATCH_W)
        h_px   = int(roi_h_p * PATCH_H)
        cv2.rectangle(frame, (x,y), (x+w_px, y+h_px), color, 2)
        cx, cy = int(traj_x[i]) if i < len(traj_x) else x+w_px//2, \
                 int(traj_y[i]) if i < len(traj_y) else y+h_px//2
        cv2.circle(frame, (cx, cy), 5, color, -1)

        bar_x = frame_w - 25
        bar_h = int(frame_h * d)
        cv2.rectangle(frame, (bar_x, frame_h-bar_h), (bar_x+15, frame_h), color, -1)
        cv2.rectangle(frame, (bar_x, 0), (bar_x+15, frame_h), (80,80,80), 1)

        put_label(frame, f"Z: {d:.2f}",                (10, 25), scale=0.6, color=color)
        put_label(frame, f"Sim: {similarities[i]:.3f}", (10, 50), scale=0.6, color=(200,200,200), thickness=1)
        put_label(frame, f"x:{cx} y:{cy}",             (10, 75), scale=0.5, color=(180,180,180), thickness=1)

        dm_viz   = (depth_maps[i] * 255).astype(np.uint8)
        dm_color = cv2.applyColorMap(dm_viz, cv2.COLORMAP_PLASMA)
        dm_small = cv2.resize(dm_color, (160, 90))
        frame[frame_h-90:frame_h, 0:160] = dm_small
        put_label(frame, "depth", (5, frame_h-93), scale=0.4, color=(200,200,200), thickness=1)

        writer.write(frame)
    writer.release()
    print(f"  ✓ {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Étape 9 — DeepSORT + YOLO
# ─────────────────────────────────────────────────────────────────────────────

def run_deepsort(frames_bgr, depth_maps, N, fps_out, output_path, frame_w, frame_h):
    from ultralytics import YOLO
    from deep_sort_realtime.deepsort_tracker import DeepSort

    detector = YOLO("yolov8n.pt")
    tracker  = DeepSort(max_age=30, n_init=3)
    writer   = make_writer(output_path, fps_out, frame_w, frame_h)
    print(f"  Tracking DeepSORT sur {N} frames...")

    track_histories = {}
    all_tracks      = {}

    for i, frame in enumerate(frames_bgr):
        annotated = frame.copy()
        results   = detector(frame, conf=0.3, verbose=False)[0]
        detections = []
        for box in results.boxes:
            x1,y1,x2,y2 = (int(v) for v in box.xyxy[0].tolist())
            conf   = float(box.conf[0])
            cls_id = int(box.cls[0])
            detections.append(([x1, y1, x2-x1, y2-y1], conf, cls_id))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            tid  = track.track_id
            ltrb = track.to_ltrb()
            x1,y1,x2,y2 = (int(v) for v in ltrb)
            cx,cy = (x1+x2)//2, (y1+y2)//2
            color = id_to_color(tid)

            bx1=max(0,x1); bx2=min(frame_w,x2)
            by1=max(0,y1); by2=min(frame_h,y2)
            dv = float(depth_maps[i][by1:by2, bx1:bx2].mean()) if bx2>bx1 and by2>by1 else 0.0

            if tid not in track_histories:
                track_histories[tid] = []
                all_tracks[tid] = {"frames":[], "depths":[], "boxes":[]}
            track_histories[tid].append((cx,cy))
            all_tracks[tid]["frames"].append(i)
            all_tracks[tid]["depths"].append(dv)
            all_tracks[tid]["boxes"].append((x1,y1,x2,y2))

            cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
            put_label(annotated, f"ID:{tid} z:{dv:.2f}", (x1, y1-8), scale=0.5, color=color)

            hist = track_histories[tid]
            for k in range(max(1, len(hist)-30), len(hist)):
                alpha = (k - (len(hist)-30)) / 30
                cv2.circle(annotated, hist[k], int(2+alpha*4), color, -1)

        dm_small = cv2.resize(cv2.applyColorMap((depth_maps[i]*255).astype(np.uint8), cv2.COLORMAP_PLASMA), (160,90))
        annotated[frame_h-90:, :160] = dm_small
        n_active = sum(1 for t in tracks if t.is_confirmed())
        put_label(annotated, f"Objets : {n_active}", (10,25))
        writer.write(annotated)
        if (i+1) % 30 == 0:
            print(f"    {i+1}/{N} | tracks actifs : {n_active}")

    writer.release()
    print(f"  ✓ {output_path}")
    return all_tracks


# ─────────────────────────────────────────────────────────────────────────────
# Étape 10 — Trajectoire 3D (ViT + MiDaS) + export HTML Plotly
# ─────────────────────────────────────────────────────────────────────────────

def build_vit_trajectory(track_positions, depth_maps, rx, ry, rw, rh,
                          roi_w_p, roi_h_p, PATCH_W, PATCH_H, N):
    traj_x = np.array([p[0] + int(roi_w_p*PATCH_W)//2 for p in track_positions], dtype=np.float32)
    traj_y = np.array([p[1] + int(roi_h_p*PATCH_H)//2 for p in track_positions], dtype=np.float32)
    traj_z = np.array([float(depth_maps[i][ry:ry+rh, rx:rx+rw].mean()) for i in range(N)], dtype=np.float32)
    wl = min(11, N if N%2==1 else N-1)
    if wl >= 3:
        traj_x = savgol_filter(traj_x, wl, 2)
        traj_y = savgol_filter(traj_y, wl, 2)
        traj_z = savgol_filter(traj_z, wl, 2)
    return traj_x, traj_y, traj_z


def build_raft_trajectory(flows_raft, depth_maps, rx, ry, rw, rh,
                           stride, N, fps, frame_w, frame_h):
    cx0 = rx + rw/2; cy0 = ry + rh/2
    traj_x = [cx0]; traj_y = [cy0]
    traj_z = [float(depth_maps[0][ry:ry+rh, rx:rx+rw].mean())]
    cur_x, cur_y = cx0, cy0

    for i, flow in enumerate(flows_raft):
        x1=int(max(0,cur_x-rw/2)); y1=int(max(0,cur_y-rh/2))
        x2=int(min(frame_w,cur_x+rw/2)); y2=int(min(frame_h,cur_y+rh/2))
        roi_flow = flow[y1:y2, x1:x2]
        if roi_flow.size == 0:
            traj_x.append(traj_x[-1]); traj_y.append(traj_y[-1]); traj_z.append(traj_z[-1])
            continue
        cur_x += float(np.mean(roi_flow[...,0]))
        cur_y += float(np.mean(roi_flow[...,1]))
        nx1=int(max(0,cur_x-rw/2)); ny1=int(max(0,cur_y-rh/2))
        nx2=int(min(frame_w,cur_x+rw/2)); ny2=int(min(frame_h,cur_y+rh/2))
        nfi = min(i*stride+stride, N-1)
        z = float(depth_maps[nfi][ny1:ny2, nx1:nx2].mean()) if nx2>nx1 and ny2>ny1 else traj_z[-1]
        traj_x.append(cur_x); traj_y.append(cur_y); traj_z.append(z)

    traj_x = np.array(traj_x, dtype=np.float32)
    traj_y = np.array(traj_y, dtype=np.float32)
    traj_z = np.array(traj_z, dtype=np.float32)
    n = len(traj_x)
    wl = min(11, n if n%2==1 else max(3,n-1))
    if wl >= 3:
        traj_x = savgol_filter(traj_x, wl, 2)
        traj_y = savgol_filter(traj_y, wl, 2)
        traj_z = savgol_filter(traj_z, wl, 2)
    return traj_x, traj_y, traj_z


def save_plotly_3d(traj_x, traj_y, traj_z, fps, output_html, title):
    times = np.arange(len(traj_x)) / fps
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=traj_x.tolist(), y=traj_z.tolist(), z=traj_y.tolist(),
        mode="lines+markers",
        line=dict(color=times.tolist(), colorscale="Plasma", width=4,
                  colorbar=dict(title="Temps (s)", thickness=12)),
        marker=dict(size=2, color=times.tolist(), colorscale="Plasma"),
        text=[f"t={t:.2f}s | x={x:.0f} | z={z:.3f}" for t,x,z in zip(times,traj_x,traj_z)],
        hoverinfo="text", name="Trajectoire"
    ))
    for idx, label, color in [(0,"Départ","lime"), (-1,"Arrivée","red")]:
        fig.add_trace(go.Scatter3d(
            x=[traj_x[idx]], y=[traj_z[idx]], z=[traj_y[idx]],
            mode="markers+text", marker=dict(size=8, color=color),
            text=[label], textposition="top center", name=label
        ))
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (px)", yaxis_title="Z (profondeur)", zaxis_title="Y (px)",
            zaxis=dict(autorange="reversed"),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        width=860, height=600
    )
    fig.write_html(str(output_html))
    print(f"  ✓ {output_html}")


# ─────────────────────────────────────────────────────────────────────────────
# Étape 11 — Fusion features (RAFT + MiDaS + ViT) par objet DeepSORT
# ─────────────────────────────────────────────────────────────────────────────

def save_fusion_plot(all_tracks, flows_raft, change_signal, stride, N, fps, output_path):
    fused = {}
    for tid, data in all_tracks.items():
        raft_in_box = []; vit_change = []
        for k, fi in enumerate(data["frames"]):
            x1,y1,x2,y2 = data["boxes"][k]
            bx1=max(0,x1); bx2=min(9999,x2); by1=max(0,y1); by2=min(9999,y2)
            ri = fi // stride
            if ri < len(flows_raft):
                f   = flows_raft[ri][by1:by2, bx1:bx2]
                mag = np.sqrt(f[...,0]**2 + f[...,1]**2)
                raft_in_box.append(float(np.mean(mag)) if mag.size > 0 else 0.0)
            else:
                raft_in_box.append(0.0)
            vc = float(change_signal[fi]) if fi < len(change_signal) else 0.0
            vit_change.append(vc)
        fused[tid] = {
            "frames"    : data["frames"],
            "depths"    : np.array(data["depths"], dtype=np.float32),
            "raft_mag"  : np.array(raft_in_box,   dtype=np.float32),
            "vit_change": np.array(vit_change,     dtype=np.float32),
        }

    n_tracks = len(fused)
    if n_tracks == 0:
        return
    fig, axes = plt.subplots(n_tracks, 3,
        figsize=(14, 3*max(n_tracks,1)), squeeze=False)
    for row, (tid, d) in enumerate(fused.items()):
        t = np.array(d["frames"]) / fps
        c = [x/255 for x in id_to_color(tid)[::-1]]
        axes[row,0].plot(t, d["raft_mag"],   color=c)
        axes[row,0].set_title(f"ID {tid} — RAFT magnitude",   fontsize=8)
        axes[row,1].plot(t, d["depths"],    color=c)
        axes[row,1].set_title(f"ID {tid} — Profondeur MiDaS", fontsize=8)
        axes[row,2].plot(t, d["vit_change"], color=c)
        axes[row,2].set_title(f"ID {tid} — Changement ViT",   fontsize=8)
        for ax in axes[row]: ax.set_xlabel("Temps (s)", fontsize=7)
    plt.suptitle("Fusion RAFT + MiDaS + ViT par objet DeepSORT", fontsize=12)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=130)
    plt.close()
    print(f"  ✓ {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    VIDEO_PATH  = Path(args.video)
    OUTPUT_DIR  = Path(args.output_dir)
    FRAME_W     = args.width
    FRAME_H     = args.height
    MAX_FRAMES  = args.max_frames
    TARGET_DIFF = args.target_diff
    FPS_OUT     = args.fps_out
    # ── Sélection du device ───────────────────────────────────────────────
    if args.device:
        DEVICE = torch.device(args.device)
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    USE_AMP = args.amp and DEVICE.type == "cuda"

    if DEVICE.type == "cuda":
        # cudnn.benchmark : autotuning des kernels convolutifs → +10-30%
        torch.backends.cudnn.benchmark = True
        # déterministe désactivé pour la vitesse max
        torch.backends.cudnn.deterministic = False
        gpu_name = torch.cuda.get_device_name(DEVICE)
        gpu_mem  = torch.cuda.get_device_properties(DEVICE).total_memory / 1e9
        print(f"  GPU     : {gpu_name} ({gpu_mem:.1f} GB)")
        if USE_AMP:
            print("  AMP     : float16 activé")
        # Préchauffage du GPU (évite la latence sur la 1ère inférence)
        _dummy = torch.zeros(1, device=DEVICE)
        del _dummy
        torch.cuda.empty_cache()

    use_raft     = not args.no_raft
    use_vit      = not (args.no_vit  or args.no_tracking)
    use_midas    = not args.no_midas
    use_deepsort = not (args.no_deepsort or args.no_tracking)

    assert VIDEO_PATH.exists(), f"Vidéo introuvable : {VIDEO_PATH}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Vidéo   : {VIDEO_PATH}")
    print(f"  Device  : {DEVICE}")
    print(f"  AMP     : {'activé' if USE_AMP else 'désactivé'}")
    print(f"  Sortie  : {OUTPUT_DIR}")
    print(f"  RAFT={use_raft} | ViT={use_vit} | MiDaS={use_midas} | DeepSORT={use_deepsort}")
    print(f"{'='*60}\n")
    print("[1/9] Chargement vidéo...")
    frames_bgr, frames_rgb, FPS, N = load_video(VIDEO_PATH, MAX_FRAMES, FRAME_W, FRAME_H)

    # ── Stride adaptatif ──────────────────────────────────────────────────
    print("\n[2/9] Stride adaptatif...")
    stride, diffs_all = compute_adaptive_stride(frames_bgr, FPS, N, TARGET_DIFF)

    # ── Farneback (toujours activé — rapide, pas de dépendance) ───────────
    print("\n[3/9] Flux optique Farneback...")
    flows_farneback, signal_farneback = compute_farneback_flows(frames_bgr, stride, N)
    make_side_by_side_video(
        frames_bgr, flows_farneback, stride, N,
        OUTPUT_DIR / "farneback_side_by_side.mp4",
        FPS_OUT, FRAME_W, FRAME_H, label_right="Farneback Flow"
    )

    # ── RAFT ──────────────────────────────────────────────────────────────
    flows_raft   = None
    signal_raft  = None
    if use_raft:
        print("\n[4/9] RAFT...")
        raft_model, InputPadder = load_raft(args.raft_dir, args.raft_weights, DEVICE)
        flows_raft, signal_raft = compute_raft_flows(
            frames_bgr, stride, N, DEVICE, raft_model, InputPadder, FRAME_W, FRAME_H, USE_AMP
        )
        make_side_by_side_video(
            frames_bgr, flows_raft, stride, N,
            OUTPUT_DIR / "raft_side_by_side.mp4",
            FPS_OUT, FRAME_W, FRAME_H, label_right="RAFT Optical Flow"
        )
    else:
        print("\n[4/9] RAFT désactivé — on utilise Farneback comme proxy.")
        flows_raft  = flows_farneback
        signal_raft = signal_farneback

    # ── ViT features ──────────────────────────────────────────────────────
    cls_tokens    = None
    patch_tokens  = None
    change_signal = None
    if use_vit:
        print("\n[5/9] ViT features...")
        vit_model, processor = load_vit(DEVICE)
        cls_tokens, patch_tokens, change_signal = extract_all_vit_features(
            frames_bgr, N, DEVICE, vit_model, processor, args.vit_batch, USE_AMP
        )
    else:
        print("\n[5/9] ViT désactivé.")

    # ── MiDaS ─────────────────────────────────────────────────────────────
    depth_maps = None
    if use_midas:
        print("\n[6/9] MiDaS profondeur...")
        midas_model, midas_transform = load_midas(DEVICE)
        depth_maps = compute_depth_maps(frames_bgr, N, DEVICE, midas_model, midas_transform, FRAME_W, FRAME_H, args.midas_batch, USE_AMP)
    else:
        print("\n[6/9] MiDaS désactivé — profondeur remplacée par 0.5.")
        depth_maps = np.full((N, FRAME_H, FRAME_W), 0.5, dtype=np.float32)

    # ── ROI + Tracking ViT ────────────────────────────────────────────────
    rx, ry, rw, rh = auto_roi(FRAME_W, FRAME_H)
    print(f"\n  ROI automatique (centre) : x={rx} y={ry} w={rw} h={rh}")

    if use_vit and patch_tokens is not None:
        print("\n[7/9] Tracking ViT...")
        track_positions, similarities, roi_w_p, roi_h_p, PATCH_W_px, PATCH_H_px = vit_track(
            patch_tokens, N, rx, ry, rw, rh, FRAME_W, FRAME_H
        )
        traj_x_vit, traj_y_vit, traj_z_vit = build_vit_trajectory(
            track_positions, depth_maps, rx, ry, rw, rh,
            roi_w_p, roi_h_p, PATCH_W_px, PATCH_H_px, N
        )
        print("\n[8/9] Vidéo overlay 3D (ViT + MiDaS)...")
        make_3d_overlay_video(
            frames_bgr, track_positions, similarities, depth_maps,
            traj_x_vit, traj_y_vit, traj_z_vit,
            roi_w_p, roi_h_p, PATCH_W_px, PATCH_H_px,
            N, FPS_OUT, OUTPUT_DIR / "resultat_3d.mp4", FRAME_W, FRAME_H
        )
        save_plotly_3d(traj_x_vit, traj_y_vit, traj_z_vit, FPS,
                       OUTPUT_DIR / "trajectoire_3d.html",
                       "Trajectoire pseudo-3D — ViT (x,y) + MiDaS (z)")
    else:
        print("\n[7/9] Tracking ViT désactivé.")
        print("\n[8/9] Vidéo overlay 3D désactivée.")

    # ── Trajectoire RAFT + MiDaS ──────────────────────────────────────────
    if flows_raft is not None and depth_maps is not None:
        traj_x_r, traj_y_r, traj_z_r = build_raft_trajectory(
            flows_raft, depth_maps, rx, ry, rw, rh, stride, N, FPS, FRAME_W, FRAME_H
        )
        save_plotly_3d(traj_x_r, traj_y_r, traj_z_r, FPS,
                       OUTPUT_DIR / "trajectoire_raft_midas.html",
                       "Trajectoire 3D — RAFT (x,y) + MiDaS (z)")

    # ── DeepSORT + YOLO ───────────────────────────────────────────────────
    all_tracks = {}
    if use_deepsort:
        print("\n[9/9] DeepSORT + YOLO...")
        all_tracks = run_deepsort(
            frames_bgr, depth_maps, N, FPS_OUT,
            OUTPUT_DIR / "resultat_deepsort.mp4", FRAME_W, FRAME_H
        )
        if use_vit and change_signal is not None and flows_raft is not None and all_tracks:
            save_fusion_plot(
                all_tracks, flows_raft, change_signal, stride, N, FPS,
                OUTPUT_DIR / "fusion_features.png"
            )
    else:
        print("\n[9/9] DeepSORT désactivé.")

    # ── Résumé ───────────────────────────────────────────────────────────
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
        used_mem  = torch.cuda.max_memory_allocated(DEVICE) / 1e9
        print(f"\n  VRAM pic utilisée : {used_mem:.2f} GB")
    print(f"\n{'='*60}")
    print("  TERMINÉ — fichiers générés :")
    for f in sorted(OUTPUT_DIR.iterdir()):
        size_mb = f.stat().st_size / 1e6
        print(f"    {f.name:<40} {size_mb:.1f} MB")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
