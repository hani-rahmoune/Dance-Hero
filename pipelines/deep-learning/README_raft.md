# Analyse vidéo — RAFT + Deep Learning

Pipeline d'analyse du mouvement combinant flux optique deep learning (RAFT), Vision Transformer (ViT), estimation de profondeur monoculaire (MiDaS) et tracking multi-objets (DeepSORT + YOLOv8).

---

## Objectif

Extraire une trajectoire pseudo-3D et des métriques de mouvement sémantiques depuis une vidéo, en fusionnant quatre sources d'information complémentaires : mouvement pixel-précis (RAFT), compréhension visuelle globale (ViT), profondeur estimée (MiDaS) et identité des objets (DeepSORT).

---

## Prérequis

### Installation des packages

```bash
pip install torch torchvision transformers plotly deep-sort-realtime ultralytics opencv-python numpy matplotlib scipy
```

### Clonage de RAFT

RAFT n'est pas sur PyPI — il faut cloner le dépôt et télécharger les poids :

```bash
git clone https://github.com/princeton-vl/RAFT.git
cd RAFT && bash download_models.sh
```

Le notebook ajoute automatiquement `RAFT/core` au `sys.path`. Placer le dossier `RAFT/` dans le même répertoire que le notebook.

### Téléchargements automatiques au premier lancement

| Modèle | Taille | Source |
|--------|--------|--------|
| ViT-Base/16 | ~330 MB | HuggingFace (`google/vit-base-patch16-224`) |
| MiDaS DPT_Hybrid | ~400 MB | `torch.hub` (intel-isl/MiDaS) |
| YOLOv8n | ~6 MB | `ultralytics` |

### GPU recommandé

Un GPU CUDA accélère significativement RAFT, ViT et MiDaS. Le notebook détecte automatiquement le device disponible (`cuda` ou `cpu`).

---

## Structure du notebook

### Cellule 1 — Configuration & imports

Définit les **hyperparamètres globaux** :

| Paramètre | Valeur par défaut | Rôle |
|-----------|------------------|------|
| `VIDEO_PATH` | `ex4.mp4` | Chemin vers la vidéo source |
| `OUTPUT_DIR` | `outputs/` | Dossier pour tous les fichiers générés |
| `FRAME_W / FRAME_H` | `640 × 360` | Résolution de travail |
| `DEVICE` | auto (`cuda` / `cpu`) | Device PyTorch |

---

### Cellule 2 — Chargement vidéo & extraction des frames

- Lit les métadonnées (FPS, résolution, durée).
- Extrait jusqu'à `MAX_FRAMES = 300` frames redimensionnées.
- Construit deux représentations :
  - `frames_bgr` : images uint8 pour OpenCV.
  - `frames_tensor` : tenseur `(N, C, H, W)` float32 pour les modèles.
- Affiche la première frame dans le notebook.

> Modifier `MAX_FRAMES` pour analyser toute la vidéo (`None`) ou la limiter.

---

### Cellule 3 — Stride adaptatif automatique

Choisit automatiquement l'espacement entre les frames soumises à RAFT pour équilibrer **qualité du flux / temps de calcul**.

**Algorithme** :
1. Calcule le diff L1 inter-frames (changement de pixels en niveaux de gris).
2. Choisit `AUTO_STRIDE` tel que le changement entre `frame[i]` et `frame[i + stride]` soit proche de `TARGET_DIFF = 3.0`.
3. Clamp entre 1 et 8 (au-delà, le flux optique devient peu fiable).

**Visualisation** : diff L1 brut vs diff lissé avec le stride choisi.

> Ajuster `TARGET_DIFF` selon la dynamique de la vidéo : `1.5` (mouvement lent) → `8.0` (mouvement rapide).

---

### Cellule 4 — RAFT : chargement & calcul du flux

**Architecture** : réseau itératif à corrélation (GRU) — plus précis que Farneback sur les grands déplacements et les zones texturées.

- Charge `raft-sintel.pth` (poids entraînés sur le dataset Sintel).
- Calcule le flux sur toutes les paires `(frame[i], frame[i + AUTO_STRIDE])`.
- Produit `signal_raft` : magnitude moyenne du flux par paire.
- **Visualisation HSV** : direction encodée en couleur sur 5 frames réparties uniformément.

> **Paramètre** : `iters=20` — nombre d'itérations de raffinement GRU (réduire pour accélérer).

---

### Cellule 5 — Export vidéo RAFT side-by-side

Génère `raft_side_by_side.mp4` :
- **Gauche** : frame originale.
- **Droite** : visualisation HSV du flux RAFT (direction = couleur, intensité = luminosité).

Labels "Original" et "RAFT Optical Flow" incrustés sur chaque côté.

---

### Cellule 6 — Comparaison RAFT vs Farneback

Recalcule Farneback sur les **mêmes paires de frames** (avec `AUTO_STRIDE`) pour une comparaison équitable.

**Métriques** :
- Visualisation HSV côte à côte sur la frame centrale.
- Signaux temporels normalisés superposés.
- Différence des signaux (zones de divergence).
- Corrélation globale entre les deux méthodes.

> Une corrélation proche de 1 indique la même structure temporelle ; les différences reflètent la précision spatiale de RAFT.

---

### Cellule 7 — ViT : extraction de features visuelles

**Modèle** : `google/vit-base-patch16-224` — Vision Transformer découpant l'image en patches 16×16.

Pour chaque frame, extrait :
- **CLS token** `(768,)` : représentation globale de la frame.
- **Patch tokens** `(196, 768)` : features locales pour chacun des 14×14 patches.

**Signal dérivé** : similarité cosinus entre CLS tokens consécutifs → `change_signal = 1 − similarité` (haut = fort changement sémantique).

**Visualisation** : comparaison RAFT (pixel) vs ViT (sémantique) normalisés.

---

### Cellule 8 — Tracking par similarité ViT

Suit une ROI sélectionnée par sa **signature sémantique** plutôt que par ses pixels.

**Principe** :
1. Sélection interactive de la ROI sur la première frame.
2. Extraction des features des patches dans la ROI → `ref_feats (768,)`.
3. Pour chaque frame suivante : recherche de la fenêtre de patches maximisant la similarité cosinus avec `ref_feats`.

**Sorties** : `track_positions` (x, y en pixels), `similarities` (évolution de la confiance).

> **Résolution** : le tracking est limité à la grille des patches (≈ 46×26 px par patch dans la résolution de travail).

---

### Cellule 9 — MiDaS : estimation de profondeur monoculaire

**Modèle** : `DPT_Hybrid` — réseau dense-prediction transformer entraîné sur des données mixtes.

- Produit une carte de profondeur **relative** normalisée [0, 1] par frame.
  - Valeur haute → proche caméra (avant-plan).
  - Valeur basse → loin caméra (arrière-plan).
- Calcule la **profondeur moyenne dans la ROI trackée** → proxy de la distance z de l'objet.

**Visualisation** : 4 frames réparties avec leur carte de profondeur (colormap `plasma`).

---

### Cellule 10 — Reconstruction pseudo-3D + overlay vidéo

Combine le tracker ViT (x, y) et MiDaS (z) pour une trajectoire 3D :

- **Lissage** Savitzky-Golay sur les 3 composantes.
- **Matplotlib 3D** : trajectoire colorée par le temps (colormap plasma).
- **Overlay vidéo** `outputs/resultat_3d.mp4` avec :
  - Rectangle coloré selon la profondeur (colormap plasma).
  - Traînée des 40 dernières positions (`TRAIL_LEN = 40`).
  - Barre de profondeur verticale (droite de l'écran).
  - Miniature de la carte de profondeur (coin bas-gauche).

---

### Cellule 11 — Trajectoire 3D interactive (Plotly)

Visualisation interactive de la trajectoire pseudo-3D :
- Rotation / zoom / survol (hover avec temps, x, y, z).
- Couleur encodant le temps (plasma).
- **Animation** frame par frame rejouable (Play / Pause).

**Sorties** :
- `outputs/trajectoire_3d.html` : visualisation interactive portable (ouvrable dans n'importe quel navigateur).

---

### Cellule 12 — DeepSORT : tracking multi-objets

Combine **YOLOv8n** (détecteur) et **DeepSORT** (tracker avec réidentification par apparence).

**Avantages de DeepSORT** :
- Conserve le même ID même après occultation (jusqu'à `max_age = 30` frames).
- Un objet n'est confirmé qu'après `n_init = 3` détections consécutives.

Pour chaque objet tracké :
- Rectangle coloré par ID.
- Profondeur MiDaS dans la bounding box.
- Traînée des 30 dernières positions.

**Sorties** :
- `outputs/resultat_deepsort.mp4` : vidéo annotée multi-objets.
- `all_tracks` : dictionnaire `{id: {frames, depths, boxes}}`.
- Résumé textuel : durée de vie, profondeur moyenne, plage temporelle de chaque objet.

---

### Cellule 13 — Fusion RAFT + MiDaS + ViT par objet DeepSORT

Pour chaque objet tracké, construit un vecteur de features temporelles fusionné :

| Feature | Source | Signification |
|---------|--------|---------------|
| `raft_mag` | RAFT dans la bbox | Intensité du mouvement pixel-précis |
| `depths` | MiDaS dans la bbox | Position z relative |
| `vit_change` | Changement ViT | Variation sémantique globale |

**Visualisation** : 3 graphes temporels par objet (RAFT · MiDaS · ViT).

**Sortie** : `outputs/fusion_features.png`.

---

### Cellule 14 — Trajectoire 3D RAFT + MiDaS (intégration du flux)

Construit une trajectoire 3D en **accumulant les déplacements RAFT** dans la ROI frame par frame :

- `x, y` : position absolue obtenue par intégration de `(dx_mean, dy_mean)`.
- `z` : profondeur MiDaS au centroïde de la ROI déplacée.

Visualisation Plotly interactive identique à la cellule 11.

**Sorties** :
- `outputs/trajectoire_raft_midas.html` : trajectoire RAFT+MiDaS interactive.

---

## Fichiers générés

| Fichier | Contenu |
|---------|---------|
| `raft_side_by_side.mp4` | Original \| RAFT HSV côte à côte |
| `outputs/resultat_3d.mp4` | Overlay trajectoire 3D ViT+MiDaS |
| `outputs/resultat_deepsort.mp4` | Tracking multi-objets DeepSORT |
| `outputs/trajectoire_3d.html` | Trajectoire 3D interactive (ViT+MiDaS) |
| `outputs/trajectoire_raft_midas.html` | Trajectoire 3D interactive (RAFT+MiDaS) |
| `outputs/fusion_features.png` | Features fusionnées par objet |

---

## Notes techniques

- **RAFT** : les poids `raft-sintel.pth` sont entraînés sur des données synthétiques (mouvement fluide). Pour des scènes très différentes, essayer `raft-kitti.pth` (scènes automobiles).
- **MiDaS** : la profondeur est **relative** — les valeurs ne correspondent pas à des mètres réels. Les comparaisons entre frames sont valides, pas les valeurs absolues.
- **ViT** : le tracking par patches est limité à une résolution de ~46 px (FRAME_W / 14). Il est robuste aux changements d'éclairage mais moins précis que CSRT pour les petits objets.
- **Mémoire** : `MAX_FRAMES = 300` charge environ 300 × 640 × 360 × 3 ≈ 200 MB en RAM. Réduire si nécessaire.
