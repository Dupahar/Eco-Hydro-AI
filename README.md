# ECO-HYDRO-AI

### Physics-Informed Digital Twin for Rural Flood Intelligence and Resilient Drainage Design

[Python](https://python.org/) [PyTorch](https://pytorch.org/) [SWMM](https://www.epa.gov/water-research/storm-water-management-model-swmm) [JavaFX](https://openjfx.io/)

Eco-Hydro-AI transforms high-resolution village drone data into a drainage-aware digital twin that can reconstruct bare-earth terrain, predict water flow through realistic settlement pathways, and produce engineering-grade visual evidence for resilient rural infrastructure planning.

---

## ⚡ Quick Results

| Metric | Value | Meaning |
| --- | --- | --- |
| Terrain input scale | 3-5 cm GSD | Village-scale terrain fidelity from drone surveys |
| Target geography | Dense Abadi settlements | Built for narrow lanes, raised plinths, and cluttered rural terrain |
| Core flood engine | Physics-informed Flood GNN surrogate | Faster than repeated full numerical what-if exploration |
| Terrain intelligence | Semantic bare-earth reconstruction | Separates true ground from buildings and obstructions |
| Output modes | Flood maps, velocity views, GIS-ready outputs, cinematic renders | Usable for demos and planning workflows |
| Strategic fit | SVAMITVA + drainage planning | Converts surveyed terrain into actionable hydrology intelligence |

---

## 🌍 What Is Eco-Hydro-AI?

Eco-Hydro-AI is a project by Team Dupahar focused on one practical challenge: chronic waterlogging in India's Abadi villages.

The system combines drone LiDAR, orthophotos, graph-based hydrology, and physics-informed AI to answer the questions conventional tools usually miss:

- Where will water actually accumulate inside the village?
- Which corridors become inaccessible first?
- Which drainage paths are physically viable?
- How do we stop routing water through houses and plinths instead of through alleys and natural flow paths?

This repository is the project archive for that system: code, outputs, experiments, submission package, and visualization evidence.

---

## ❗ Problem Statement

India has already generated large volumes of high-resolution rural survey data, but village drainage decisions still rely on incomplete slope intelligence.

Key failure points:

- Conventional terrain products are too coarse for micro-topography.
- Buildings and elevated plinths are often mistaken for true ground.
- Water routing becomes unrealistic in dense settlements.
- Drainage systems are built without physics-validated local flow behavior.

The result is repeated monsoon waterlogging, mobility collapse, infrastructure damage, and avoidable redesign cost.

---

## 🎯 Objectives

- Reconstruct true bare-earth terrain beneath cluttered village scenes.
- Build flood prediction that respects mass conservation and drainage topology.
- Visualize flood risk in a way that engineers, judges, and planners can inspect immediately.
- Convert existing drone survey assets into decision-ready drainage intelligence.

---

## 🧭 End-to-End System Overview

1. High-resolution village terrain and orthophoto acquisition
2. Semantic terrain synthesis to remove geometric shortcut errors
3. Bare-earth DTM generation with building-aware correction
4. SWMM / hydrology-ground-truth aligned graph construction
5. Physics-informed flood surrogate inference
6. Export of flood depth, velocity, risk zones, and visual outputs

```text
Drone LiDAR + Orthophoto
          |
          v
Semantic Terrain Synthesis
          |
          v
Bare-Earth Digital Terrain Model
          |
          v
Hydrology Graph + Physics Constraints
          |
          v
Flood GNN Surrogate Prediction
          |
          v
4D Visualizations + GIS Outputs + Drainage Insight
```

---

## 🧠 Model and Simulation Strategy

### Terrain Intelligence

- Uses a semantic terrain synthesis approach inspired by Sonata-style representation learning.
- Learns to distinguish plinths, buildings, vegetation, and temporary clutter from actual terrain.
- Produces bare-earth terrain suitable for gravity-aware drainage design.

### Hydrology Intelligence

- Uses graph-based flood modeling over village terrain.
- Applies physics-informed behavior instead of unrestricted black-box inference.
- Enforces realistic routing behavior by suppressing conductivity across building barriers.
- Focuses on depth, movement, and accessibility rather than only binary flood labels.

### Why This Matters

- Rural lanes are narrow.
- Building footprints distort naive flow models.
- Small vertical errors can completely flip drainage direction.
- The model needs to be visually convincing and physically defensible.

---

## 📈 Performance Snapshot

- Designed for centimeter-aware settlement terrain reasoning.
- Built to support rapid scenario exploration without repeatedly running slow end-to-end hydraulic workflows.
- Structured to preserve physically plausible flood behavior through topology-aware graph modeling.
- Positioned for drainage lifecycle savings by reducing slope-design errors and downstream redesign.

---

## 📸 The Visual Evidence

This project has real output artifacts, not just architecture diagrams.

### 1. Flyover Animation

The original flyover GIF is too large for reliable GitHub inline playback in a README.

[Open the Dariyapur flyover animation](DTM_Flood_Prediction_Final_Submission/5_Visualizations/Animations/Flyover_118118%20Dariyapur.gif)

Preview image:

<p align="center">
  <img src="DTM_Flood_Prediction_Final_Submission/5_Visualizations/Animations/dariyapurr.jpeg" alt="Dariyapur animation preview" width="900" />
</p>

### 2. Rendered Output Gallery

<p align="center">
  <img src="DTM_Flood_Prediction_Final_Submission/5_Visualizations/4K_Renders/Cinematic_4K_118118%20Dariyapur.png" alt="Cinematic 4K render" width="32%" />
  <img src="DTM_Flood_Prediction_Final_Submission/5_Visualizations/4K_Renders/Velocity_Field_118118%20Dariyapur.png" alt="Velocity field render" width="32%" />
  <img src="DTM_Flood_Prediction_Final_Submission/5_Visualizations/Animations/dariyapurr.jpeg" alt="GIS output" width="32%" />
</p>

From left to right:

- cinematic flood render
- velocity field visualization
- GIS-style output preview

### 3. Why These Outputs Matter

- The cinematic render proves presentation-grade spatial coherence.
- The velocity map shows that the system is modeling movement, not just static accumulation.
- The GIS preview shows downstream planning compatibility.

---

## 📂 Repository Contents

### 1. Submission Package

`DTM_Flood_Prediction_Final_Submission/`

Contains the structured submission material:

- executive summary
- code and models
- documentation
- results and metrics
- visualizations
- presentation assets
- datasets

### 2. GodTier Workspace

`GodTier/`

Contains the heavier engineering workspace with:

- training code
- visualization pipelines
- Java visualization application
- reports and metrics
- generated outputs

### 3. Data Assets

- `Point-Cloud/`
- `Training Data Set ORI & SHP File/`
- `OutPut Testing ORI/`

These directories hold source and derived assets used during experimentation and presentation.

---

## 🚀 How To Explore This Repo

1. Start with `DTM_Flood_Prediction_Final_Submission/README.txt`
2. Open the visual outputs in `DTM_Flood_Prediction_Final_Submission/5_Visualizations/`
3. Review `GodTier/README.md`
4. Inspect the training and inference code inside `GodTier/`
5. Use the rendered images and reports to understand the end-to-end story quickly

---

## 🗺️ Deployment and Impact

- Supports drainage planning in dense rural settlements.
- Converts already collected survey data into usable hydrology intelligence.
- Helps planners validate whether water is routed around structures realistically.
- Enables communication through visuals that non-technical stakeholders can understand.
- Fits naturally into village-scale resilience and infrastructure decision workflows.

---

## 🔭 Current Limitation and Next Step

Current limitation:

- The repository still contains large visual and dataset artifacts mixed with code and submission material.

Recommended next step:

- Split the public-facing demo, core models, and private/heavy datasets into cleaner repositories for maintenance and scale.

---

## 🧱 Project Character

This is not a toy notebook project.

It is a combined:

- AI terrain reconstruction workflow
- physics-informed flood modeling effort
- digital twin visualization system
- hackathon / submission package
- engineering evidence archive

That is why the repository feels dense: it is part lab, part product demo, and part decision-support artifact.

---

## Team Dupahar

**Autonomous Digital Terrain Synthesis and Resilient Drainage Optimization** is built around a simple principle:

**make hidden terrain visible, make flood behavior explainable, and make drainage design defensible.**