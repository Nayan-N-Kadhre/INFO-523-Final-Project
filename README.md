
[Colab Notebook](https://colab.research.google.com/drive/1bmnIhLuadExK_7jOSf6q7KckCHoj9OFd#scrollTo=OHLCX2kCVe3u)

# Real Estate Pricing & Socioeconomic Analysis
INFO 523 · Team 4 — Nayan Kadhre & Erik Jensen

Analyzing U.S. housing affordability by combining real estate listings, household income, and cost-of-living data. Uses regression, classification, and geospatial clustering to explore where homeownership is viable for average Americans.

---

## Datasets

All datasets sourced from Kaggle. No personally identifiable information — IRB review not required.

| Dataset | Records | Features |
|---|---|---|
| USA Real Estate | 2,226,382 | 12 |
| US Household Income Statistics | 32,527 | 19 |
| US Cost of Living | 3,171 | 9 |

## Research Questions

1. Is homeownership probable based on city-level salaries and living expenses?
2. How is wealth distributed geographically, and does it align with other economic indicators?

## Methods

- **Regression** — Linear, random forest, gradient boosting to predict affordability
- **Classification** — Logistic regression, decision trees, random forest to label cities as affordable or not
- **Geospatial analysis** — Spatial clustering and heatmaps for regional insights
- **Association rule mining** — Apriori algorithm to surface co-occurring economic patterns

## Tools & Libraries

`R` · `tidyverse` · `ggplot2` · `arules` · `glm` · `rpart` · `lm` · `Google Colab` · `GitHub`

## Project Structure

```
├── data/               # Raw and processed datasets
├── notebooks/          # R scripts and Colab notebooks
├── results/            # Output figures and model results
└── README.md
```

## Expected Outcomes

- Identify regions where homeownership is attainable on average wages
- Quantify the gap between income growth and rising housing costs
- Provide evidence on whether salary is keeping pace with cost of living across the U.S.

---

*University of Arizona · INFO 523 Data Mining and Discovery · Spring 2025*
