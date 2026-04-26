---
title: "pv-cleaning-multimodal-semantic-slam: Multi-modal semantic SLAM and uncertainty-aware navigation for photovoltaic cleaning robots"
tags:
  - Python
  - SLAM
  - semantic mapping
  - uncertainty
  - robotics
authors:
  - name: Ryrant
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 2026-04-26
bibliography: paper.bib
---

# Summary

`pv-cleaning-multimodal-semantic-slam` is a Python software package that reproduces a task-oriented SLAM and navigation workflow for photovoltaic cleaning robots. It combines visual-inertial and LiDAR-inertial uncertainty fusion, semantic occupancy mapping, and uncertainty-aware path planning in a single reproducible simulation.

# Statement of need

Photovoltaic farms are difficult for conventional localization systems because they are repetitive, reflective, and geometrically degenerate. For cleaning robots, localization accuracy alone is insufficient; the system must also reason about panel boundaries, safe traversal, and repeated-cleaning penalties. This package provides:

1. A reproducible baseline for multi-modal pose fusion under modality degradation.
2. Probabilistic semantic mapping suitable for cleaning task planning.
3. A navigation cost formulation that integrates path efficiency, boundary safety, and localization uncertainty.
4. A compact experimentation scaffold for comparing ablation variants.

The software supports method prototyping, educational demonstration, and benchmark-style evaluation for field robotics in photovoltaic maintenance.

# Implementation

The implementation includes:

- Synthetic photovoltaic environment generation with panel strips, gaps, and obstacles.
- Gaussian fusion between simulated VIO and LIO state estimates.
- Semantic grid map with log-odds updates and cleaned-area tracking.
- Uncertainty score estimation from covariance and modality quality indicators.
- Risk-aware planner with terms for boundary risk, uncertainty, and repeated-coverage cost.

The executable workflow outputs task coverage, uncertainty statistics, trajectory quality proxies, and safety margin indicators.

# Acknowledgements

This implementation follows foundational approaches from visual SLAM, visual-inertial estimation, LiDAR odometry, and probabilistic robotics [@mur2017orbslam2; @qin2018vinsmono; @zhang2014loam; @thrun2005probabilistic].

