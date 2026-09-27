# Final Project
My Graduation project for my M.Sc. in Data science @HIT

# TL;DR
A comparative deep-learning study for breast-cancer detection in digital breast tomosynthesis, replicating a published clinical pipeline on real hospital data and extending it across five modern vision architectures. Trained on ~35,500 annotated slices from 784 studies, evaluated on a held-out set of 343 cases, with a three-level inference design — per-slice scoring, sliding-window aggregation to a scan, and view fusion to a case. The best model reaches 0.999 case-level AUC (0.984 sensitivity / 0.991 specificity), and the study additionally used Stable Diffusion to synthesise supplementary studies for dataset enrichment.

Breast cancer Background
-Second most common malignancy in women worldwide
-One of eight women will be diagnosed in their lifetime
-Early diagnosis critical for reducing mortality
Current Screening Methods:
1. Full Field Digital Mammography (FFDM)
2. Digital Breast Tomosynthesis (DBT)
3. Breast MRI for high-risk cases


Project’s Objectives
- Enhance performance by using additional deep learning architectures on DBT images.
- By synthetically enriching the dataset, the goal is to increase both specificity and sensitivity.

The first phase of the project consisted of preprocessing of the data. In the second phase, we've applied a few transformer architucures on the DBT dataset(with two different resolutions) to enhance today's best classification results. On the third phase, we've enriched the DB with images of a minority kind (dense breasts and calicifications, which the model tend to miss more than any other kind) using GenAI, then, re-running our models once again.

