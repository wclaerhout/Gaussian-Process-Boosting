# Gaussian Process Boosting (GPBoost) Discovery

This repository explores the integration of **Gaussian Processes (GP)** with **Gradient Boosting Decision Trees (GBDT)**, specifically focusing on the **GPBoost** framework. The project aims to understand how structured variance—such as individual differences in biological data—can be accounted for in predictive modeling. This repo applies the ideas of the paper of Sigrist (2022).

## Project Motivation
In many real-world datasets, observations are not truly independent. For instance, multiple samples might come from the same individual, introducing a "grouping" effect or "random effect." Standard boosting algorithms (like XGBoost or LightGBM) often struggle with this grouped structure because they assume independence between rows.

This repository documents an experiment using synthetic data to compare:
1.  **Naive Boosting:** How well does a standard GBDT perform when ignoring the grouping variable?
2.  **GPBoost:** How does a model that explicitly accounts for structured variance perform, especially when predicting for individuals not seen during training?

## The Experiment: Cortisol Level Prediction
The synthetic dataset simulates **Cortisol levels** based on:
-   **Typical Parameters:** Continuous and categorical features (e.g., time of day, stress markers, age).
-   **The Hidden Factor:** A grouping variable representing the **Donor (Person)**. Each donor has a unique baseline shift or "random effect" that influences their cortisol levels beyond the global features.

### Phase 1: Naive Boosting
We first train a standard boosting model on the synthetic data. 
-   **The Challenge:** The model is not given the donor ID. 
-   **Goal:** To establish a baseline and see how much error is introduced by "unexplained" variance that is actually tied to the individual donor.

### Phase 2: GPBoost & Structured Variance
We then utilize the **GPBoost** library. This approach allows us to combine the non-linear mapping capabilities of GBDTs with a Gaussian Process to model the grouped structure.

-   **Grouped Random Effects:** GPBoost accounts for the specific variance attributed to each donor.
-   **Extrapolation via Kernels:** A key interest is the use of the **RBF (Radial Basis Function) Kernel**. This allows the model to make informed predictions even for *new persons* not present in the training set, by leveraging similarity in the latent space.

## Key Goals
-   [ ] Generate synthetic longitudinal/grouped cortisol data.
-   [ ] Benchmark LightGBM/XGBoost against the grouped data.
-   [ ] Implement GPBoost with a grouping factor.
-   [ ] Evaluate performance on "unseen" donors to test the GP kernel's generalization.

## References
-   Sigrist, F. (2022). [GPBoost Python Package](https://github.com/pfriedrich/GPBoost).
-   Sigrist, F. (2020). "Gaussian Process Boosting". *Journal of Machine Learning Research*.
