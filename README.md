# tiktok-virality-triadic-framework

> **Reproducibility package for** *"Prevalence is not advantage: A reproducible audit of content features, K-pop salience, and the limits of predicting algorithmically mediated visibility on TikTok"*

---

## Citation

### APA

Chansanam, W., Jitsaeng, K., Chaikhambunga, J., Junlabuddee, S., Hoaihongthong, S., Wisassinthu, B., Sutthiprapa, K., & Nguyen, L. T. (2027). Prevalence is not advantage: A reproducible audit of content features, K-pop salience, and the limits of predicting algorithmically mediated visibility on TikTok. *International Journal of Data and Network Science*, *11*. https://doi.org/10.5267/j.ijdns.2026.10.012

### BibTeX

```bibtex
@article{chansanam2027prevalence,
  author    = {Chansanam, Wirapong and Jitsaeng, Kanokwan and Chaikhambunga, Jirarat
               and Junlabuddee, Siriporn and Hoaihongthong, Suwanna
               and Wisassinthu, Benjawan and Sutthiprapa, Kornkanok
               and Nguyen, Lan Thi},
  title     = {Prevalence is not advantage: {A} reproducible audit of content features,
               {K}-pop salience, and the limits of predicting algorithmically mediated
               visibility on {TikTok}},
  journal   = {International Journal of Data and Network Science},
  volume    = {11},
  year      = {2027},
  doi       = {10.5267/j.ijdns.2026.10.012},
  note      = {Article ID: ijdns\_2026\_158.
               Received: June 17, 2026; Revised: September 25, 2026;
               Accepted: October 20, 2026; Online: October 20, 2026.
               ISSN 2561-8156 (Online) · 2561-8148 (Print)},
  url       = {https://doi.org/10.5267/j.ijdns.2026.10.012}
}
```

---

## Overview

This repository contains the complete reproducibility package for the study described above. The study examines whether K-pop content on TikTok enjoys a *visibility advantage* beyond mere *prevalence* — that is, whether K-pop videos systematically outperform non-K-pop videos in algorithmically mediated metrics (views, likes, shares, comments) once confounding content features are controlled.

The analytic sample comprises **93 TikTok videos** collected via a Diversity-Representative Threshold (DRT) sampling protocol (Figure 1) from a United States endpoint. Six negative binomial regression models (M0–M4 + robustness check) consistently returned **null results** for K-pop salience as a predictor of engagement metrics, while surface-level algorithmic hashtag use also showed no reliable advantage. The study foregrounds reproducibility, transparent AI-assisted coding with vocabulary-adherence auditing, and open data practices.

---

## Repository Contents

| File / Folder | Description |
|---|---|
| `analysis_revision.py` | Complete analysis script reproducing all statistics, tables, and figures in the manuscript |
| `analytic_dataset_derived.csv` | Derived analytic dataset with all engineered features (93 rows); creator handles withheld per privacy policy |
| `results_revision.json` | Machine-readable results (all descriptive and inferential statistics) |
| `models_final.json` | Model outputs for all six models: M0 (intercept-only), M1–M4 (feature blocks), and robustness check |
| `ai_coding_audit.json` | LLM coding audit results including vocabulary-adherence rates and inter-coder agreement |
| `coding_prompt.txt` | Verbatim Gemini 1.5 Pro coding prompt used in the AI-assisted content analysis pipeline |
| `n8n_workflow_content_analyzer.json` | Exported n8n workflow (27 nodes) implementing the retrieval and AI coding pipeline |
| `appendix_a_hashtag_classification.py` | Appendix A: hashtag classification dictionary (algorithmic / niche / descriptive) and caption regular expressions |
| `figures/` | Manuscript figures: Figure 1 (DRT protocol), Figure 2 (sample flow), Figure 3 (prevalence vs. advantage) |

---

## Reproducibility

### Requirements

| Package | Version |
|---|---|
| Python | 3.9 |
| pandas | 2.3.3 |
| scikit-learn | 1.6.1 |
| statsmodels | latest compatible |
| SciPy | 1.13.1 |

All stochastic operations use **random seed 42**.

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/wirapong/tiktok-virality-triadic-framework.git
cd tiktok-virality-triadic-framework

# 2. Install dependencies
pip install pandas==2.3.3 scikit-learn==1.6.1 scipy==1.13.1 statsmodels

# 3. Run the analysis
python analysis_revision.py
```

> **Note:** `analysis_revision.py` reads from `analytic_dataset_derived.csv` and writes outputs to `results_revision.json`, `models_final.json`, and the `figures/` directory. The raw video files and original scraper output are **not redistributed** (see Data Availability below). The script is fully self-contained given the derived CSV.

---

## Data Availability

- **Included:** Public TikTok video identifiers, aggregate engagement metrics (views, likes, shares, comments), and all engineered content features used in the models.
- **Excluded — video files:** Raw TikTok video files are not redistributed in compliance with TikTok's Terms of Service.
- **Excluded — creator handles:** Creator usernames/handles have been withheld from the released dataset to protect individual privacy, consistent with the ethical framework of the study.
- **Collection method:** Data were collected via [Apify](https://apify.com/) using the `apidojo/tiktok-scraper` actor, United States geographic endpoint, during the data collection window specified in the manuscript.
- **AI coding pipeline:** The n8n workflow (`n8n_workflow_content_analyzer.json`) and verbatim prompt (`coding_prompt.txt`) are provided to enable full replication of the LLM-assisted content coding procedure using Gemini 1.5 Pro.

---

## Ethics & Limitations

All data were collected from publicly accessible TikTok content. No user authentication, private data, or platform API developer credentials were used beyond the publicly available scraper endpoint. Creator handles are withheld from the released dataset. The study's scope is limited to 93 videos sampled from a single geographic endpoint (United States) over a bounded collection window; findings should not be generalised to other regions, time periods, or platform algorithm states without replication.

---

## License

This repository is released under the [MIT License](LICENSE).

```
MIT License

Copyright (c) 2026 Wirapong Chansanam et al.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
