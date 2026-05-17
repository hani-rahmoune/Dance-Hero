# Analyse vidéo — Optical Flow classique

Pipeline d'analyse du mouvement basé sur des algorithmes classiques de flux optique (OpenCV), sans modèle deep learning.

---

## Objectif

Extraire des métriques de mouvement depuis une vidéo (répétitions, cadence, phases, asymétrie) à partir d'une ou plusieurs ROI sélectionnées manuellement.

---

## Prérequis

```bash
pip install opencv-python numpy matplotlib scipy pandas
```

Placer la vidéo source dans le même répertoire que le notebook et renseigner `VIDEO_PATH` dans la cellule 1.

---

## Structure du notebook

### Cellule 1 — Configuration & imports

Définit les **hyperparamètres centralisés** :

| Paramètre | Valeur par défaut | Rôle |
|-----------|------------------|------|
| `VIDEO_PATH` | `exemple2.mp4` | Chemin vers la vidéo source |
| `OUTPUT_PATH` | `resultat.mp4` | Chemin de la vidéo de sortie |
| `FRAME_W / FRAME_H` | `640 × 360` | Résolution de travail |
| `TRACKER_TYPE` | `CSRT` | Algorithme de tracking (`CSRT` ou `KCF`) |

---

### Cellule 2 — Chargement vidéo

- Lit les métadonnées natives (FPS, résolution, durée).
- Extrait et redimensionne la première frame.
- Affiche un aperçu dans le notebook.

---

### Cellule 3 — Sélection de la ROI

Ouvre une fenêtre interactive OpenCV pour dessiner le rectangle de suivi.

> **Contrôles** : cliquer-glisser → **Espace/Entrée** pour valider · **C** pour annuler.

Retourne un tuple `(x, y, w, h)` en pixels dans la résolution de travail.

---

### Cellule 4 — Initialisation du tracker

Instancie le tracker sélectionné (`CSRT` recommandé pour les membres du corps, `KCF` pour la vitesse) et l'initialise sur la première frame + ROI.

---

### Cellule 5 — Boucle principale : Farneback + tracking

Pour chaque frame :

1. **Tracker** : prédit la nouvelle position de la ROI.
2. **Optical flow dense** (Farneback) : calcule un vecteur `(dx, dy)` pour chaque pixel.
3. **Signal** : extrait la magnitude moyenne dans la ROI → `signal_raw`.

**Sorties** :
- `signal_raw` : liste brute des magnitudes par frame.
- `frames_output.csv` : index des frames (optionnel, `SAVE_CSV = True`).

> **Paramètres Farneback** : `pyr_scale=0.5 · levels=3 · winsize=15 · iterations=3 · poly_n=5 · poly_sigma=1.2`

---

### Cellule 6 — Analyse du signal & détection des répétitions

- **Lissage** : filtre Savitzky-Golay (`window=11, polyorder=2`).
- **Détection des pics** : `scipy.find_peaks` avec seuil adaptatif (`mean + 0.5 × std`).
- **Métriques calculées** :
  - Nombre de répétitions
  - Cadence (reps/min)
  - Durée moyenne par répétition
- **Visualisation** : signal brut + lissé + pics + durée par rep (indicateur de fatigue).

---

### Cellule 7 — Rendu vidéo annoté

Génère `OUTPUT_PATH` avec :
- Vecteurs de flux optique (flèches vertes) dans la ROI.
- Rectangle du tracker (vert = OK, rouge = perdu).
- Indicateur **BEAT** sur les frames correspondant aux pics.
- Graphe du signal en incrustation (coin bas-droit).
- Compteur de répétitions en temps réel.

---

### Cellule 8 — Lucas-Kanade sparse (comparaison)

Alternative à Farneback : suit uniquement ~100 **points d'intérêt Shi-Tomasi** dans la ROI.

- Plus propre et plus rapide que le flux dense.
- Fournit en plus les composantes **`dx` / `dy`** séparées (direction du mouvement).
- Re-détecte les points toutes les `REDETECT_EVERY = 15` frames.
- **Export** : `optical_flow_lk.csv` avec colonnes `frame, mag, dx, dy`.
- **Visualisation** : comparaison Farneback vs Lucas-Kanade superposés.

---

### Cellule 9 — Analyse des phases (concentrique / excentrique)

Exploite le signe de `dy` moyen pour segmenter le mouvement :

| `dy` | Direction | Phase |
|------|-----------|-------|
| `< 0` | ↑ vers le haut | Concentrique |
| `> 0` | ↓ vers le bas | Excentrique |

> En OpenCV, `y = 0` est en **haut** de l'image.

Calcule la durée moyenne de chaque phase et visualise les transitions.

---

### Cellule 10 — Visualisation HSV du flux optique

Encode la direction du flux dans la couleur (espace HSV) :

- **Teinte** → direction du vecteur (0°=rouge, 120°=vert, 240°=bleu)
- **Valeur** → magnitude (noir = immobile, vif = rapide)
- **Saturation** → 255 (couleur pure)

Génère `resultat_couleurs.mp4` en side-by-side : `Original | Flux HSV`.

---

### Cellule 11 — Heatmap d'activité

Accumule la magnitude du flux pixel par pixel sur toute la vidéo → carte de chaleur des zones les plus actives.

**Sorties** :
- `heatmap_mouvement.png` : heatmap brute + superposition sur la première frame.

Utile pour détecter des **compensations** : zones qui bougent alors qu'elles ne devraient pas.

---

### Cellule 12 — Multi-ROI & corrélation croisée

Permet d'analyser `N_ROIS = 2` zones indépendantes (ex : bras gauche / bras droit).

**Métriques** :
- Corrélation croisée normalisée entre chaque paire de signaux.
- Lag au pic de corrélation (décalage temporel en secondes).
- Asymétrie instantanée (différence des magnitudes).

| Corrélation | Interprétation |
|-------------|----------------|
| ≈ 1 | Mouvements synchrones |
| ≈ 0 | Mouvements indépendants |
| < 0 | Mouvements opposés (alterné) |

---

### Cellule 13 — Background Subtraction & trajectoire du centre de masse

Utilise **MOG2** pour séparer l'avant-plan du fond sans ROI manuelle.

- Calcule le **centroïde** du masque foreground → trajectoire 2D automatique.
- Génère `resultat_bgs.mp4` en side-by-side : `Trajectoire | Foreground`.
- **Visualisation** : trajectoire 2D · coordonnées x/y temporelles · pixels foreground normalisés.

> **Paramètres MOG2** : `history=100 · varThreshold=50 · detectShadows=False`

---

### Cellule 14 — Dashboard final

Regroupe toutes les métriques dans un dashboard matplotlib 3×4 :

- Signal Farneback vs Lucas-Kanade
- Phases concentrique/excentrique
- Durée par répétition (rouge = fatigue)
- Heatmap accumulée
- Trajectoire 2D centre de masse
- Carte de métriques texte
- Intensité globale (BGS)
- Asymétrie multi-ROI

**Sortie** : `dashboard_final.png` (150 dpi).

---

## Fichiers générés

| Fichier | Contenu |
|---------|---------|
| `resultat.mp4` | Vidéo annotée avec flux optique et compteur |
| `resultat_couleurs.mp4` | Side-by-side original \| flux HSV |
| `resultat_bgs.mp4` | Side-by-side trajectoire \| foreground |
| `frames_output.csv` | Index des frames analysées |
| `optical_flow_lk.csv` | Signal LK : `frame, mag, dx, dy` |
| `heatmap_mouvement.png` | Carte de chaleur d'activité |
| `dashboard_final.png` | Dashboard synthèse toutes métriques |
