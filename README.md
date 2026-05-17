#  Dance Hero — Extraction automatique du rythme depuis une vidéo

> Détection de mouvements répétitifs à partir de vidéos silencieuses, sans audio.  
> Projet de recherche — L3 Informatique, Université d'Angers (2025/2026)

---

## C'est quoi ?

On transforme une vidéo en signal temporel de mouvement, puis on détecte automatiquement les événements rythmiques : sauts, rotations, répétitions d'exercice, mouvements de danse. Pas d'audio. Juste la vidéo.

Le projet contient deux pipelines indépendants et un benchmark quantitatif sur 5 vidéos annotées manuellement.

---

## Structure du repo

```
dance-hero/
│
├── pipelines/
│   ├── optical-flow/
│   │   ├── analyse_video_opticalflow.py          # Script standalone
│   │   ├── analyse_video_opticalflow_notebook.ipynb
│   │   └── README.md
│   │
│   └── deep-learning/
│       ├── analyse_video_raft_notebook.ipynb     # RAFT + ViT + MiDaS + DeepSORT
│       ├── analyse_video_.py                     # Script standalone
│       ├── README.md
│
├── benchmark/
│   ├── rhythm_benchmark.py                       # Pipeline universel (7 méthodes)
│   ├── annotator.py                              # Outil d'annotation vérité terrain
│   │
│   ├── floss/
│   │   ├── ground_truth.csv
│   │   └── benchmark_results/
│       ├── notebook.ipynb   
│   ├── jumping-rope/
│   ├── shika-dance/
│   ├── spin/
│   └── zombie-dance/
│
├── biceps-curls-demo/
│   ├── counting_reps_with_neural_networks.ipynb  # Comparaison modèles de pose
│   └── biceps_curls.mp4
├── requirements.txt
└── README.md
```

---

## Les deux pipelines

### 1. Optical Flow classique (`pipelines/optical-flow/`)

Farnebäck, Lucas-Kanade, CSRT/KCF, MOG2. Pas de GPU nécessaire.

```bash
pip install opencv-python numpy matplotlib scipy pandas
python pipelines/optical-flow/analyse_video_opticalflow.py --video ma_video.mp4
```

### 2. Deep Learning (`pipelines/deep-learning/`)

RAFT, ViT, MiDaS, DeepSORT. GPU recommandé.

```bash
# Cloner RAFT (pas sur PyPI)
git clone https://github.com/princeton-vl/RAFT.git pipelines/deep-learning/RAFT
cd pipelines/deep-learning/RAFT && bash download_models.sh

pip install torch torchvision transformers plotly deep-sort-realtime ultralytics
python pipelines/deep-learning/analyse_video_.py --video ma_video.mp4
```

---

## Benchmark

7 méthodes comparées sur 5 vidéos annotées frame par frame.

| Méthode | Floss | Zombie | Jumping | Shika | Spin | **Moy.** |
|---|---|---|---|---|---|---|
| RAFT | 1,00 | 1,00 | 0,80 | 0,62 | 0,67 | **0,82** |
| DTW | 0,60 | 0,95 | 0,91 | 0,27 | 0,63 | 0,67 |
| Dense Flow | 0,93 | 0,64 | 0,96 | 0,18 | 0,29 | 0,60 |
| Sparse LK | 0,80 | 0,64 | 0,96 | 0,08 | 0,48 | 0,59 |
| Pose MediaPipe | 1,00 | 0,45 | 0,73 | 0,35 | 0,32 | 0,57 |
| Pixel Difference | 0,55 | 0,27 | 0,75 | 0,56 | 0,20 | 0,47 |
| Dense Points LK | 0,19 | 0,18 | 0,96 | 0,17 | 0,20 | 0,34 |

**Ce qu'on retient :** RAFT est le plus robuste globalement. Sparse LK fait aussi bien sur les mouvements simples en étant 15x plus rapide.

Lancer le benchmark sur une nouvelle vidéo :

```python
# Dans rhythm_benchmark.py
VIDEO_PATH  = "ma_video.mp4"
GT_PATH     = "ground_truth.csv"
MOTION_TYPE = "impact"       # fast_cyclic | slow_cyclic | impact | lateral
REGION      = "lower"        # full | upper | lower | head
```

---

## Annoter sa propre vérité terrain

```bash
python benchmark/annotator.py --video ma_video.mp4 --output ground_truth.csv
```

Navigation frame par frame → `SPACE` pour marquer un pic → exporte `ground_truth.csv` avec les colonnes `event_id`, `frame_peak`, `time_sec`.

---

## Installation complète

```bash
pip install -r requirements.txt
```

Pour PyTorch avec CUDA :
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

RAFT (non disponible sur PyPI) :
```bash
git clone https://github.com/princeton-vl/RAFT.git pipelines/deep-learning/RAFT
cd pipelines/deep-learning/RAFT && bash download_models.sh
```

---

## Environnement

- Python 3.12
- CUDA 11.8+ recommandé (RAFT, ViT, MiDaS)
- Testé sur Windows + NVIDIA RTX 2050 et Google Colab

---

## Auteurs

Hani Rahmoune · Terbouche Amine · Mayas Khemri  
Encadrant : Benoît Da Mota — Université d'Angers, Faculté des Sciences
