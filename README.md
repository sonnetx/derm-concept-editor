# Concept Removal Analysis

Scripts to measure the effect of visual concept removal on model predictions and internal activations.

## Overview

These scripts use manual image editing to test and validate concept ranking methods. After training a concept model, we evaluate the hypothesis that **removing a visual concept from an image leads to reduced activation of the corresponding internal concept representation**.

We demonstrate this method with the ruler concept specifically, testing whether:
1. Removing rulers from images changes p(malignant) predictions
2. This change correlates with reduced activation of the internal "ruler" concept

## Scripts

### `concept_removal_analysis.py`

- Trains a concept model to map activations to CLIP concept space
- Computes model predictions (p(malignant)) for both original and edited images
- Extracts internal concept activations for all concepts, but specifically focuses the target concept (e.g., "ruler")

**Contributions:**
- Full concept decomposition of internal activations
- Identifies which concepts are most affected by the removal
- Correlates activation changes with prediction changes

**Outputs:**
- `summary.json` - Key statistics and findings
- `results.csv` - Per-image detailed results
- `score_differences.png` - Distribution of prediction changes
- `score_comparison.png` - Before/after scatter plot
- `ruler_activation_diff.png` - Distribution of ruler activation changes
- `ruler_activation_comparison.png` - Before/after activation scatter
- `ruler_correlation.png` - Correlation between activation and prediction changes
- `top_affected_concepts.png` - Which concepts changed most

## Score Types

### 1. `malignant_prob`
- Directly measures p("malignant" token)
- Used to understand absolute prediction changes
- Higher = more confident in malignancy

### 2. `contrastive`
- Measures log p("malignant") - log p("benign")
- Used to understand relative preference between classes
- Higher = stronger preference for malignant