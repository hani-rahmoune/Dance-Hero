# Dance Hero — Extraction automatique du rythme depuis une vidéo

> Détection de mouvements répétitifs à partir de vidéos silencieuses, sans audio.  
> Projet de recherche — L3 Informatique, Université d'Angers (2025/2026)

---

## C'est quoi ?

On transforme une vidéo en signal temporel de mouvement, puis on détecte automatiquement les événements rythmiques dedans : sauts, rotations, répétitions d'exercice, mouvements de danse.

Pas d'audio. Juste la vidéo.

---

## Méthodes comparées

| Méthode | Type | Vitesse | F1 moyen |
|---|---|---|---|
| RAFT | Deep learning (flux optique) | Lent (~18s) | **0,82** |
| DTW | Similarité temporelle | Lent (~30s) | 0,67 |
| Dense Flow (Farnebäck) | Classique | Moyen | 0,60 |
| Sparse LK | Classique | **Rapide (~1s)** | 0,59 |
| Pose MediaPipe | Landmarks corporels | Moyen | 0,57 |
| Pixel Difference | Classique | **Très rapide** | 0,47 |
| Dense Points LK | Classique | Rapide | 0,34 |
| ViT Features | Deep learning (features) | Moyen | 0,83* |
| MiDaS Depth | Deep learning (profondeur) | Moyen | 0,42* |

*\* évalués uniquement sur Zombie Dance (sémantique `motion_energy`)*

---

## Structure du projet

```
.
├── optical_flow_version_plus.ipynb   # Pipeline classique (Farnebäck, LK, MOG2)
├── test_stride.ipynb                 # Pipeline deep learning (RAFT, ViT, MiDaS, DeepSORT)
├── counting_reps_with_neural_networks.ipynb  # Comparaison modèles de pose (biceps curls)
├── annotator.py                      # Outil d'annotation vérité terrain
├── rhythm_benchmark.py               # Benchmark universel
│
├── floss/
│   ├── ground_truth.csv
│   ├── video.mp4
│   └── benchmark_results/
├── jumping rope/
├── shika dance/
├── spin/
└── zombie dance/
```

---

## Résultats en un coup d'œil

Le F1-score par vidéo (les méthodes sont triées par moyenne décroissante) :

| Méthode | Floss | Zombie | Jumping | Shika | Spin | **Moy.** |
|---|---|---|---|---|---|---|
| RAFT | 1,00 | 1,00 | 0,80 | 0,62 | 0,67 | **0,82** |
| DTW | 0,60 | 0,95 | 0,91 | 0,27 | 0,63 | 0,67 |
| Dense Flow | 0,93 | 0,64 | 0,96 | 0,18 | 0,29 | 0,60 |
| Sparse LK | 0,80 | 0,64 | 0,96 | 0,08 | 0,48 | 0,59 |
| Pose MediaPipe | 1,00 | 0,45 | 0,73 | 0,35 | 0,32 | 0,57 |
| Pixel Difference | 0,55 | 0,27 | 0,75 | 0,56 | 0,20 | 0,47 |
| Dense Points LK | 0,19 | 0,18 | 0,96 | 0,17 | 0,20 | 0,34 |

**Ce qu'on retient :** RAFT est le plus robuste globalement, mais Sparse LK fait aussi bien sur les mouvements simples en étant 15x plus rapide.

---

## Lancer le benchmark

```bash
pip install mediapipe torch torchvision timm opencv-python scipy scikit-learn
```

Ouvrir `rhythm_benchmark.py` et configurer :

```python
VIDEO_PATH    = "mon_video.mp4"
GT_PATH       = "ground_truth.csv"
EVENT_LABEL   = "jump"        # spin | beat | rep | jump | floss ...
MOTION_TYPE   = "impact"      # fast_cyclic | slow_cyclic | impact | lateral
REGION        = "lower"       # full | upper | lower | head
```

Puis lancer. Les résultats (CSV + graphiques) s'exportent automatiquement dans `benchmark_results/`.

---

## Annoter sa propre vérité terrain

```bash
# Dans un notebook Colab, ouvrir annotator.py
# Charger la vidéo → naviguer frame par frame → SPACE pour marquer un pic → exporter CSV
```

L'outil génère un fichier `ground_truth.csv` avec trois colonnes : `event_id`, `frame_peak`, `time_sec`.

---

## Pourquoi ViT et MiDaS sont absents de certaines vidéos

Ces deux méthodes ne fonctionnent qu'avec la sémantique `motion_energy`. Sur les vidéos configurées en `position_extreme` (Floss, Jumping Rope, Shika, Spin), elles sont automatiquement ignorées par le benchmark car leurs signaux mesurent une énergie globale et non une position.

---

## Environnement

- Python 3.12
- PyTorch + CUDA (GPU recommandé pour RAFT et ViT)
- Google Colab pour les expériences lourdes
- OpenCV, MediaPipe, timm, scipy

---

## Auteurs

Hani Rahmoune · Terbouche Amine · Mayas Khemri  
Encadrant : Benoît Da Mota — Université d'Angers, Faculté des Sciences
