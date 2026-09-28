# References

Papers from `Elicit.bib` that inform TrustDrive MiniLab, grouped by the component
of the demonstrator they support. PDFs are stored locally but git-ignored
(copyright); this index is tracked.

## Downloaded (PDF in this folder)

| file | reference | relevance to the project |
| --- | --- | --- |
| `Antonante2022_Monitoring_of_Perception_Systems.pdf` | Antonante, Nilsen, Carlone, *Monitoring of Perception Systems: Deterministic, Probabilistic, and Learning-based Fault Detection and Identification*, Artificial Intelligence, 2022. [arXiv](https://arxiv.org/pdf/2205.10906) | Closest to our integrity monitor: run-time detection/identification of perception faults, including diagnostic-graph and consistency ideas behind the NIS check. |
| `Yatbaz2024_Integrity_Monitoring_3D_Object_Detection.pdf` | Yatbaz, Dianati, Koufos, Woodman, *Integrity Monitoring of 3D Object Detection in Automated Driving Systems*, IEEE ITSC 2024. [arXiv](https://arxiv.org/pdf/2405.07600) | Direct precedent for the term "integrity monitoring" of a learned perception block; introspection when the network is confidently wrong. |
| `Luo2020_Workload_Adaptive_Haptic_Shared_Control.pdf` | Luo et al., *A Workload Adaptive Haptic Shared Control Scheme for Semi-Autonomous Driving*, Accident Analysis & Prevention, 2020. [arXiv](https://arxiv.org/pdf/2004.00167) | Basis for our workload-modulated authority law `λ = Φ(I, W)` and the feed-forward-always / feedback-arbitrated blend. |
| `Monsaingeon2023_Reliability_Displays_Trust_Automation.pdf` | Monsaingeon, Caroux, Langlois, Lemercier, *Multimodal interface and reliability displays: effect on attention, mode awareness, and trust in partially automated vehicles*, Frontiers in Psychology, 2023. [PDF](https://www.frontiersin.org/articles/10.3389/fpsyg.2023.1107847/pdf) | Human-factors grounding for surfacing an integrity/reliability signal to the driver (mode awareness, calibrated trust). |
| `Chen2023_End_to_End_Autonomous_Driving_Survey.pdf` | Chen, Wu, Chitta, Jaeger, Geiger, Li, *End-to-End Autonomous Driving: Challenges and Frontiers*, IEEE TPAMI, 2023. [arXiv](https://arxiv.org/pdf/2306.16927) | Background on learned perception-to-control pipelines and their open safety/uncertainty problems (motivation section). |

## Could not auto-download

| reference | link | why not fetched |
| --- | --- | --- |
| Charmet, Berge-Cherfaoui, Guzman, Armand, *Operational Design Domain Monitoring with Uncertain Measurements*, IEEE ITSC 2024. | [hal.science/hal-04829701](https://hal.science/hal-04829701/document) | HAL is behind a Cloudflare check; open the link in a browser and save the PDF here as `Charmet2024_ODD_Monitoring_Uncertain_Measurements.pdf`. **Highly relevant** (ODD monitoring under measurement uncertainty). |

## Link-only (publisher / Semantic Scholar landing pages, no open PDF)

Relevant but without a direct open-access file in the .bib:

- Chen et al., *Uncertainty-aware Sensor Data Anomaly Detection for Autonomous Vehicles*, IEEE IV 2024 — doi:10.1109/IV55156.2024.10588587
- Sun et al., *Uncertainty Propagation in AI-driven Autonomous Systems*, YAC 2025 — doi:10.1109/YAC66630.2025.11149732
- Zhang, Wang, Song, Wen, *Integrity-Monitored Deep Reinforcement Learning for Safe and Robust Autonomous Navigation*, IEEE ITSC 2025 — doi:10.1109/ITSC60802.2025.11423028
- Guajardo et al., *Certified Perception for Autonomous Cars*, 2021 (Toyota Research Institute)
- Bürkle, Geissler, Paulitsch, Scholl, *Fault-Tolerant Perception for Automated Driving: A Lightweight Monitoring Approach*, 2021
- Tang et al., *Prediction-Uncertainty-Aware Decision-Making for Autonomous Vehicles*, IEEE T-IV 2022 — doi:10.1109/TIV.2022.3188662
- Li, Bai, Yang, Zhang, *Integrated Uncertainty-Aware Decision and Control for Autonomous Driving with Chance Constraints*, 2026 — doi:10.1145/3810280.3810352
- Izadi, Saraphis, Ghasemi, *Dynamic Control Authority Arbitration for Resolving a Reverse-Intent Conflict in a Haptic Shared Control Paradigm*, IFAC 2022 — doi:10.1016/j.ifacol.2022.11.234
- Izadi, Ghasemi, *Quantifying the Performance of an Adaptive Haptic Shared Control Paradigm for Steering a Ground-Vehicle*, 2022 — doi:10.2139/ssrn.4139413
- Zhang, Lou, Hu, Zhu, Lv, *Socially-Compliant Hierarchical Human-Vehicle Collaboration With Multimodal Haptic Steering*, IEEE T-ITS 2025 — doi:10.1109/TITS.2025.3575593
- Hubmann et al., *Automated Driving in Uncertain Environments: Planning With Interaction and Uncertain Maneuver Prediction*, IEEE T-IV 2018 — doi:10.1109/TIV.2017.2788208
