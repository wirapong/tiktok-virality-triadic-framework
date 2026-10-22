#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproducible analysis for the revised manuscript:
"Algorithmically Mediated Virality on TikTok"
Int. Journal of Data and Network Science -- Major Revision, 2026-09.

Input : ../../Reels-Grid view-20082025.xlsx   (AI-coded analytic dataset, n=93)
        ../../Reels-Grid view.csv             (wave 1 raw retrieval, n=73)
        ../../Reels-Grid view-03072025.csv    (wave 2 raw retrieval, n=93)
        ../../Content_Analyzer__Tiktok_.json  (n8n workflow provenance)
Output : results_revision.json + printed report
"""
import json, re, warnings, hashlib
import numpy as np, pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict, GroupKFold
from sklearn.metrics import (confusion_matrix, classification_report, roc_auc_score,
                             average_precision_score, precision_recall_fscore_support)
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")
RNG = 42
np.random.seed(RNG)
BASE = ".."
OUT = {}

# ----------------------------------------------------------------------------
# 1. SAMPLE-SELECTION FLOW  (Reviewer major #1)
# ----------------------------------------------------------------------------
w1 = pd.read_csv(f"{BASE}/../Reels-Grid view.csv")
w2 = pd.read_csv(f"{BASE}/../Reels-Grid view-03072025.csv")
df = pd.read_excel(f"{BASE}/../Reels-Grid view-20082025.xlsx")
df.columns = [c.strip() for c in df.columns]

flow = {}
flow["wave1_retrieved"] = len(w1)
flow["wave2_retrieved"] = len(w2)
union_ids = set(w1.reelId) | set(w2.reelId)
flow["union_unique_ids"] = len(union_ids)
flow["duplicate_records_across_waves"] = len(w1) + len(w2) - len(union_ids)
flow["coded_dataset_rows"] = len(df)
flow["unique_reelIds_final"] = int(df.reelId.nunique())
flow["duplicate_rows_in_final"] = int(len(df) - df.reelId.nunique())
flow["unique_creators"] = int(df.user.nunique())
flow["creators_with_multiple_videos"] = int((df.user.value_counts() > 1).sum())
flow["max_videos_per_creator"] = int(df.user.value_counts().max())
flow["missing_caption"] = int(df.caption.isna().sum())
flow["missing_hashtags"] = int(df.hashtags.isna().sum())
flow["missing_metrics"] = int(df[["videoViewCount", "likesCount", "commentsCount"]].isna().any(axis=1).sum())
flow["ai_coded_complete"] = int(df[["category", "style", "tone", "hook", "visualHook", "CTA"]].notna().all(axis=1).sum())

# upload timestamps
ts = pd.to_datetime(df.timestamp.str.replace("\u202f", " ", regex=False),
                    format="%B %d, %Y at %I:%M %p UTC", errors="coerce")
df["upload_dt"] = ts
flow["upload_window_start"] = str(ts.min())
flow["upload_window_end"] = str(ts.max())
flow["upload_span_days"] = float((ts.max() - ts.min()).days)
OUT["sample_flow"] = flow

# ----------------------------------------------------------------------------
# 2. FEATURE ENGINEERING -- strictly exogenous to the outcome (Reviewer #3)
# ----------------------------------------------------------------------------
df["caption"] = df.caption.fillna("")
df["hashtags"] = df.hashtags.fillna("")

ALGO = {"fyp", "fypシ", "foryou", "foryoupage", "foryourpage", "viral", "viralvideo",
        "viralvideos", "trending", "trend", "tiktokviral", "viraltiktok", "xyzbca",
        "fyppppppppppppppppppppppp", "tiktok", "explore", "explorepage", "fy"}
NICHE_PAT = re.compile(r"(fan|fans|stan|army|blink|once|carat|moa|nctzen|bunnies|"
                       r"tiny|zenz|fandom|community|tok$|lovers|nation|edit|edits|"
                       r"cover|challenge|squad|club)", re.I)

def tags(s):
    return [t.strip().lower() for t in str(s).split(",") if t.strip()]

def classify_tag(t):
    if t in ALGO:
        return "algorithmic"
    if NICHE_PAT.search(t):
        return "niche"
    return "descriptive"

rows = []
SECOND = re.compile(r"\b(you|your|yours|y'?all|u|ur)\b", re.I)
CURIO = re.compile(r"(wait for it|you won'?t believe|guess|secret|watch till|watch until|"
                   r"this is why|find out|part \d|pov|don'?t scroll|omg|\?{2,}|…|\.\.\.)", re.I)
PROBLEM = re.compile(r"(problem|struggle|hate when|tired of|why does|can'?t|stop|"
                     r"mistake|wrong|fix|help|hard|annoying|issue|no one)", re.I)
IMPER = re.compile(r"^\s*(try|watch|check|listen|look|stream|follow|comment|tag|save|do)\b", re.I)

for _, r in df.iterrows():
    tl = tags(r.hashtags)
    cls = [classify_tag(t) for t in tl]
    cap = re.sub(r"#\S+", "", str(r.caption)).strip()   # caption text w/o hashtags
    rows.append(dict(
        n_hashtags=len(tl),
        n_hashtags_unique=len(set(tl)),
        n_algo=cls.count("algorithmic"),
        n_niche=cls.count("niche"),
        n_desc=cls.count("descriptive"),
        share_algo=cls.count("algorithmic") / len(tl) if tl else 0.0,
        share_niche=cls.count("niche") / len(tl) if tl else 0.0,
        share_desc=cls.count("descriptive") / len(tl) if tl else 0.0,
        layer_count=sum(x > 0 for x in [cls.count("algorithmic"), cls.count("niche"), cls.count("descriptive")]),
        cap_chars=len(cap),
        cap_words=len(cap.split()),
        cap_question=int("?" in cap),
        cap_second_person=int(bool(SECOND.search(cap))),
        cap_intrigue=int(bool(CURIO.search(cap))),
        cap_problem=int(bool(PROBLEM.search(cap))),
        cap_imperative=int(bool(IMPER.search(cap))),
        cap_emoji=int(bool(re.search(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", cap))),
    ))
F = pd.DataFrame(rows, index=df.index)
df = pd.concat([df, F], axis=1)

# --- controlled vocabulary for genre (Reviewer #10) -------------------------
def canon_genre(s):
    s = str(s).lower().strip()
    if re.search(r"k-?\s?pop|korean pop|kpop", s):
        return "K-pop"
    if "dance" in s or "choreo" in s:
        return "Dance (non-K-pop)"
    if "music" in s:
        return "Music (non-K-pop)"
    if "entertain" in s or "comedy" in s or "skit" in s or "humor" in s:
        return "Entertainment/Comedy"
    if "cosplay" in s or "culture" in s or "social" in s or "anime" in s or "animation" in s:
        return "Culture/Fandom"
    return "Other/Lifestyle"

def canon_style(s):
    s = str(s).lower()
    if "fancam" in s or "fan cam" in s:
        return "Fancam"
    if "livestream" in s or "live " in s or "concert" in s:
        return "Live/Livestream"
    if "dance" in s or "choreo" in s or "performance" in s:
        return "Performance/Choreography"
    if "edit" in s or "compil" in s or "playlist" in s or "highlight" in s or "lyric" in s:
        return "Edit/Compilation"
    if "inform" in s or "listicle" in s or "rank" in s or "review" in s or "interview" in s or "vlog" in s:
        return "Informational/Talk"
    return "Short-form video (generic)"

def canon_tone(s):
    s = str(s).lower()
    if "energetic" in s or "excite" in s or "enthusias" in s or "high-energy" in s:
        return "High-energy"
    if "playful" in s or "fun" in s or "humor" in s or "funny" in s:
        return "Playful/Humorous"
    if "emotion" in s or "nostalg" in s or "senti" in s or "roman" in s or "calm" in s or "sooth" in s:
        return "Affective/Calm"
    if "inform" in s or "educat" in s or "serious" in s or "neutral" in s:
        return "Informational/Neutral"
    return "Engaging (unspecified)"

df["genre"] = df.category.map(canon_genre)
df["style_c"] = df.style.map(canon_style)
df["tone_c"] = df.tone.map(canon_tone)
df["is_kpop"] = (df.genre == "K-pop").astype(int)
df["is_perf"] = df.style_c.isin(["Performance/Choreography", "Fancam", "Live/Livestream"]).astype(int)
df["cta_present"] = (~df.CTA.astype(str).str.strip().str.lower()
                     .isin(["", "nan", "none", "no", "no cta", "none.", "n/a", "absent"])).astype(int)

# outcomes
df["log_views"] = np.log10(df.videoViewCount)
df["log_likes"] = np.log10(df.likesCount)
df["log_comments"] = np.log10(df.commentsCount)
df["like_rate"] = df.likesCount / df.videoViewCount
df["comment_rate"] = df.commentsCount / df.videoViewCount
p90 = df.videoViewCount.quantile(0.90)
df["viral"] = (df.videoViewCount >= p90).astype(int)
OUT["virality_threshold"] = {"p90_views": float(p90), "n_viral": int(df.viral.sum()),
                             "n_nonviral": int((1 - df.viral).sum()),
                             "prevalence": float(df.viral.mean())}

# ----------------------------------------------------------------------------
# 3. DESCRIPTIVES  (Reviewer #4, minor: distinguish metrics)
# ----------------------------------------------------------------------------
desc = {}
for c in ["videoViewCount", "likesCount", "commentsCount", "like_rate", "comment_rate",
          "n_hashtags", "cap_words"]:
    s = df[c].astype(float)
    desc[c] = dict(n=int(s.notna().sum()), mean=float(s.mean()), sd=float(s.std()),
                   median=float(s.median()), q1=float(s.quantile(.25)), q3=float(s.quantile(.75)),
                   minimum=float(s.min()), maximum=float(s.max()),
                   skew=float(stats.skew(s.dropna())), kurtosis=float(stats.kurtosis(s.dropna())),
                   iqr=float(s.quantile(.75) - s.quantile(.25)))
OUT["descriptives"] = desc

# frequency tables reconciled to N
for var, key in [("genre", "genre_freq"), ("style_c", "style_freq"),
                 ("tone_c", "tone_freq"), ("cta_present", "cta_freq")]:
    vc = df[var].value_counts()
    OUT[key] = {str(k): {"n": int(v), "pct": round(100 * v / len(df), 1)} for k, v in vc.items()}
    OUT[key]["_TOTAL"] = {"n": int(vc.sum()), "pct": 100.0}

# hashtag class totals
OUT["hashtag_composition"] = {
    "total_tag_tokens": int(df.n_hashtags.sum()),
    "mean_tags_per_video": float(df.n_hashtags.mean()),
    "sd_tags_per_video": float(df.n_hashtags.std()),
    "algorithmic_tokens": int(df.n_algo.sum()),
    "niche_tokens": int(df.n_niche.sum()),
    "descriptive_tokens": int(df.n_desc.sum()),
    "pct_algorithmic": round(100 * df.n_algo.sum() / df.n_hashtags.sum(), 1),
    "pct_niche": round(100 * df.n_niche.sum() / df.n_hashtags.sum(), 1),
    "pct_descriptive": round(100 * df.n_desc.sum() / df.n_hashtags.sum(), 1),
    "videos_using_all_three_layers": int((df.layer_count == 3).sum()),
    "videos_with_no_hashtags": int((df.n_hashtags == 0).sum()),
}

# ----------------------------------------------------------------------------
# 4. CORRELATIONS with CI  (Reviewer #4)
# ----------------------------------------------------------------------------
def r_ci(x, y, alpha=.05):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    n = len(x)
    r, p = stats.pearsonr(x, y)
    z = np.arctanh(r); se = 1 / np.sqrt(n - 3)
    lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    rs, ps = stats.spearmanr(x, y)
    return dict(n=n, r=round(float(r), 3), ci=[round(float(lo), 3), round(float(hi), 3)],
                p=float(p), rho=round(float(rs), 3), p_spearman=float(ps))

corr_vars = ["n_hashtags", "n_algo", "n_niche", "n_desc", "share_algo", "layer_count",
             "cap_words", "cap_second_person", "cap_intrigue", "cap_problem",
             "is_kpop", "is_perf", "cta_present"]
OUT["correlations_with_log_views"] = {v: r_ci(df[v], df.log_views) for v in corr_vars}
OUT["correlations_with_like_rate"] = {v: r_ci(df[v], df.like_rate) for v in corr_vars}

# Holm-Bonferroni on the primary family
pv = {k: v["p"] for k, v in OUT["correlations_with_log_views"].items()}
order = sorted(pv, key=pv.get)
m = len(order); adj = {}
prev = 0
for i, k in enumerate(order):
    a = min(1.0, (m - i) * pv[k]); a = max(a, prev); prev = a
    adj[k] = round(a, 4)
OUT["correlations_holm_adjusted_p"] = adj

# ----------------------------------------------------------------------------
# 5. PCA  (Reviewer #5)
# ----------------------------------------------------------------------------
pca_vars = ["n_hashtags", "share_algo", "share_niche", "layer_count", "cap_chars",
            "cap_words", "cap_second_person", "cap_intrigue", "is_perf", "is_kpop"]
X = df[pca_vars].astype(float).values
Xs = StandardScaler().fit_transform(X)
R = np.corrcoef(Xs, rowvar=False)

# Bartlett's test of sphericity
n, p = Xs.shape
chi2 = -(n - 1 - (2 * p + 5) / 6) * np.log(np.linalg.det(R))
dofb = p * (p - 1) / 2
bart_p = stats.chi2.sf(chi2, dofb)

# KMO
Rinv = np.linalg.pinv(R)
Q = np.zeros_like(R)
for i in range(p):
    for j in range(p):
        Q[i, j] = -Rinv[i, j] / np.sqrt(Rinv[i, i] * Rinv[j, j])
np.fill_diagonal(Q, 1)
off_r = R[~np.eye(p, dtype=bool)] ** 2
off_q = Q[~np.eye(p, dtype=bool)] ** 2
kmo = off_r.sum() / (off_r.sum() + off_q.sum())
kmo_i = {}
for i in range(p):
    idx = [j for j in range(p) if j != i]
    kmo_i[pca_vars[i]] = round(float((R[i, idx] ** 2).sum() /
                                     ((R[i, idx] ** 2).sum() + (Q[i, idx] ** 2).sum())), 3)

pca_full = PCA().fit(Xs)
eig = pca_full.explained_variance_
evr = pca_full.explained_variance_ratio_
k = int((eig > 1).sum())

# parallel analysis (Horn) -- 1000 random datasets
sim = np.zeros((1000, p))
for b in range(1000):
    Z = np.random.normal(size=(n, p))
    sim[b] = PCA().fit(StandardScaler().fit_transform(Z)).explained_variance_
pa95 = np.percentile(sim, 95, axis=0)
k_pa = int((eig > pa95).sum())

pca_k = PCA(n_components=max(k, 1)).fit(Xs)
load = pca_k.components_.T * np.sqrt(pca_k.explained_variance_)
scores = pca_k.transform(Xs)
for i in range(scores.shape[1]):
    df[f"PC{i+1}"] = scores[:, i]

OUT["pca"] = {
    "variables_entered": pca_vars,
    "n": int(n),
    "kmo_overall": round(float(kmo), 3),
    "kmo_per_item": kmo_i,
    "bartlett_chi2": round(float(chi2), 2),
    "bartlett_df": int(dofb),
    "bartlett_p": float(bart_p),
    "eigenvalues": [round(float(e), 3) for e in eig],
    "explained_variance_pct": [round(float(e) * 100, 2) for e in evr],
    "cumulative_pct": [round(float(x) * 100, 2) for x in np.cumsum(evr)],
    "parallel_analysis_95th": [round(float(x), 3) for x in pa95],
    "n_components_kaiser": k,
    "n_components_parallel": k_pa,
    "loadings": {pca_vars[i]: [round(float(load[i, j]), 3) for j in range(load.shape[1])]
                 for i in range(p)},
    "communalities": {pca_vars[i]: round(float((load[i] ** 2).sum()), 3) for i in range(p)},
}

# ----------------------------------------------------------------------------
# 6. OLS with creator-clustered robust SE + diagnostics (Reviewer #4, minor)
# ----------------------------------------------------------------------------
def ols_report(formula, data, cluster="user"):
    mod = smf.ols(formula, data=data)
    res_ols = mod.fit()
    res = mod.fit(cov_type="cluster", cov_kwds={"groups": data[cluster]})
    ci = res.conf_int()
    tab = {}
    for name in res.params.index:
        tab[name] = dict(b=round(float(res.params[name]), 4),
                         se=round(float(res.bse[name]), 4),
                         t=round(float(res.tvalues[name]), 3),
                         p=float(res.pvalues[name]),
                         ci=[round(float(ci.loc[name, 0]), 4), round(float(ci.loc[name, 1]), 4)])
    # standardized betas from the non-clustered fit
    Xd = mod.exog[:, 1:]; yd = mod.endog
    sb = {}
    for j, name in enumerate(mod.exog_names[1:]):
        sx = Xd[:, j].std()
        sb[name] = round(float(res_ols.params[name] * sx / yd.std()), 3)
    bp = het_breuschpagan(res_ols.resid, res_ols.model.exog)
    vif = {}
    for j, name in enumerate(mod.exog_names):
        if name == "Intercept":
            continue
        vif[name] = round(float(variance_inflation_factor(mod.exog, j)), 2)
    infl = res_ols.get_influence()
    cooks = infl.cooks_distance[0]
    return {
        "formula": formula, "n": int(res.nobs), "n_clusters": int(data[cluster].nunique()),
        "r2": round(float(res_ols.rsquared), 3), "adj_r2": round(float(res_ols.rsquared_adj), 3),
        "f": round(float(res_ols.fvalue), 3), "f_p": float(res_ols.f_pvalue),
        "rmse": round(float(np.sqrt(res_ols.mse_resid)), 4),
        "aic": round(float(res_ols.aic), 2), "bic": round(float(res_ols.bic), 2),
        "coefficients": tab, "standardized_beta": sb, "vif": vif,
        "breusch_pagan_LM": round(float(bp[0]), 3), "breusch_pagan_p": float(bp[1]),
        "ramsey_reset_p": float(linear_reset(res_ols, power=2, use_f=True).pvalue),
        "shapiro_resid_p": float(stats.shapiro(res_ols.resid)[1]),
        "durbin_watson": round(float(sm.stats.durbin_watson(res_ols.resid)), 3),
        "max_cooks_d": round(float(cooks.max()), 4),
        "n_cooks_above_4_over_n": int((cooks > 4 / len(cooks)).sum()),
        "condition_number": round(float(np.linalg.cond(mod.exog)), 1),
    }

M1 = ("log_views ~ share_algo + n_hashtags + layer_count + cap_words + "
      "cap_second_person + cap_intrigue + cap_problem + is_kpop + is_perf + cta_present")
OUT["model1_content_only"] = ols_report(M1, df)

M0 = "log_views ~ is_kpop + is_perf"
OUT["model0_genre_only"] = ols_report(M0, df)

npc = OUT["pca"]["n_components_kaiser"]
M2 = "log_views ~ " + " + ".join(f"PC{i+1}" for i in range(npc))
OUT["model2_pca_components"] = ols_report(M2, df)

M3 = ("like_rate ~ share_algo + n_hashtags + cap_second_person + cap_intrigue + "
      "is_kpop + is_perf + cta_present")
OUT["model3_like_rate"] = ols_report(M3, df)

M4 = ("comment_rate ~ share_algo + n_hashtags + cap_second_person + cap_intrigue + "
      "is_kpop + is_perf + cta_present")
OUT["model4_comment_rate"] = ols_report(M4, df)

# Engagement->exposure model reported SEPARATELY and flagged as non-causal
M5 = "log_views ~ log_likes + log_comments"
OUT["model5_engagement_association_NONCAUSAL"] = ols_report(M5, df)

# ----------------------------------------------------------------------------
# 7. K-POP BASELINE COMPARISON  (Reviewer #9)
# ----------------------------------------------------------------------------
kp = df[df.is_kpop == 1]; nk = df[df.is_kpop == 0]

def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))

def boot_ci_medratio(a, b, B=5000):
    out = []
    for _ in range(B):
        out.append(np.median(np.random.choice(a, len(a))) / np.median(np.random.choice(b, len(b))))
    return [round(float(np.percentile(out, 2.5)), 3), round(float(np.percentile(out, 97.5)), 3)]

kpop_block = {
    "prevalence_in_sampling_frame_pct": round(100 * df.is_kpop.mean(), 1),
    "n_kpop": int(len(kp)), "n_nonkpop": int(len(nk)),
    "kpop_share_of_top_decile_pct": round(100 * df[df.viral == 1].is_kpop.mean(), 1),
    "n_kpop_in_top_decile": int(df[df.viral == 1].is_kpop.sum()),
}
u, pu = stats.mannwhitneyu(kp.videoViewCount, nk.videoViewCount, alternative="two-sided")
kpop_block["views"] = {
    "median_kpop": float(kp.videoViewCount.median()),
    "median_nonkpop": float(nk.videoViewCount.median()),
    "median_ratio": round(float(kp.videoViewCount.median() / nk.videoViewCount.median()), 3),
    "median_ratio_boot95CI": boot_ci_medratio(kp.videoViewCount.values, nk.videoViewCount.values),
    "mannwhitney_U": float(u), "p": float(pu),
    "rank_biserial_r": round(float(1 - 2 * u / (len(kp) * len(nk))) * -1, 3),
    "cliffs_delta": round(float(cliffs_delta(kp.videoViewCount, nk.videoViewCount)), 3),
    "hedges_g_log": round(float((kp.log_views.mean() - nk.log_views.mean()) /
                                np.sqrt(((len(kp) - 1) * kp.log_views.var() + (len(nk) - 1) * nk.log_views.var()) /
                                        (len(kp) + len(nk) - 2))), 3),
}
for metric in ["like_rate", "comment_rate"]:
    u2, p2 = stats.mannwhitneyu(kp[metric], nk[metric], alternative="two-sided")
    kpop_block[metric] = {
        "median_kpop": round(float(kp[metric].median()), 5),
        "median_nonkpop": round(float(nk[metric].median()), 5),
        "mannwhitney_U": float(u2), "p": float(p2),
        "cliffs_delta": round(float(cliffs_delta(kp[metric], nk[metric])), 3),
    }
# chi-square: K-pop over/under-represented in top decile relative to frame
ct = pd.crosstab(df.is_kpop, df.viral)
chi2c, pc, dofc, exp = stats.chi2_contingency(ct)
kpop_block["kpop_x_topdecile"] = {
    "table": ct.values.tolist(),
    "chi2": round(float(chi2c), 3), "df": int(dofc), "p": float(pc),
    "fisher_p": float(stats.fisher_exact(ct.values)[1]),
    "cramers_v": round(float(np.sqrt(chi2c / len(df))), 3),
    "min_expected": round(float(exp.min()), 2),
}
OUT["kpop_baseline"] = kpop_block

# ----------------------------------------------------------------------------
# 8. CLASSIFICATION  (Reviewer #11) -- honest small-sample protocol
# ----------------------------------------------------------------------------
feat = ["share_algo", "n_hashtags", "layer_count", "cap_words", "cap_second_person",
        "cap_intrigue", "cap_problem", "is_kpop", "is_perf", "cta_present"]
Xc = df[feat].astype(float).values
yc = df.viral.values
groups = df.user.values

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
scoring = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]

models = {
    "logistic_l2_balanced": Pipeline([("sc", StandardScaler()),
                                      ("m", LogisticRegression(max_iter=5000, class_weight="balanced",
                                                               C=1.0, random_state=RNG))]),
    "random_forest_balanced": Pipeline([("m", RandomForestClassifier(n_estimators=500, min_samples_leaf=3,
                                                                     class_weight="balanced_subsample",
                                                                     random_state=RNG))]),
    "dummy_majority": Pipeline([("m", DummyClassifier(strategy="most_frequent"))]),
    "dummy_stratified": Pipeline([("m", DummyClassifier(strategy="stratified", random_state=RNG))]),
}
clf_out = {}
for name, pipe in models.items():
    cvres = cross_validate(pipe, Xc, yc, cv=cv, scoring=scoring, n_jobs=1, error_score=np.nan)
    entry = {s: {"mean": round(float(np.nanmean(cvres['test_' + s])), 3),
                 "sd": round(float(np.nanstd(cvres['test_' + s])), 3),
                 "folds": [round(float(v), 3) for v in cvres['test_' + s]]} for s in scoring}
    yp = cross_val_predict(pipe, Xc, yc, cv=cv)
    cm = confusion_matrix(yc, yp)
    pr, rc, f1, sup = precision_recall_fscore_support(yc, yp, zero_division=0)
    entry["confusion_matrix_TN_FP_FN_TP"] = [int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])]
    entry["per_class"] = {"non_viral": {"precision": round(float(pr[0]), 3), "recall": round(float(rc[0]), 3),
                                        "f1": round(float(f1[0]), 3), "support": int(sup[0])},
                          "viral": {"precision": round(float(pr[1]), 3), "recall": round(float(rc[1]), 3),
                                    "f1": round(float(f1[1]), 3), "support": int(sup[1])}}
    clf_out[name] = entry

# creator-grouped CV as leakage/dependence check
gkf = GroupKFold(n_splits=5)
g = cross_validate(models["logistic_l2_balanced"], Xc, yc, cv=gkf.split(Xc, yc, groups),
                   scoring=["balanced_accuracy", "roc_auc", "average_precision"], error_score=np.nan)
clf_out["logistic_grouped_by_creator"] = {
    s: {"mean": round(float(np.nanmean(g['test_' + s])), 3), "sd": round(float(np.nanstd(g['test_' + s])), 3)}
    for s in ["balanced_accuracy", "roc_auc", "average_precision"]}

# permutation test on balanced accuracy
from sklearn.model_selection import permutation_test_score
sc, perm, pval = permutation_test_score(models["logistic_l2_balanced"], Xc, yc, cv=cv,
                                        scoring="balanced_accuracy", n_permutations=1000,
                                        random_state=RNG, n_jobs=1)
clf_out["permutation_test"] = {"observed_balanced_accuracy": round(float(sc), 3),
                               "null_mean": round(float(perm.mean()), 3), "p": float(pval),
                               "n_permutations": 1000}
clf_out["class_distribution"] = {"viral": int(yc.sum()), "non_viral": int((1 - yc).sum()),
                                 "positive_rate": round(float(yc.mean()), 3)}
clf_out["events_per_variable"] = round(float(yc.sum() / len(feat)), 2)
clf_out["features_used"] = feat
clf_out["features_excluded_to_prevent_leakage"] = [
    "videoViewCount", "log_views", "likesCount", "commentsCount",
    "like_rate", "comment_rate", "view_to_interaction_ratio"]
OUT["classification"] = clf_out

# regression CV for continuous exposure (log views)
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score, KFold
reg = Pipeline([("sc", StandardScaler()), ("m", RidgeCV(alphas=np.logspace(-3, 3, 25)))])
kf = KFold(5, shuffle=True, random_state=RNG)
r2cv = cross_val_score(reg, Xc, df.log_views.values, cv=kf, scoring="r2")
rmsecv = -cross_val_score(reg, Xc, df.log_views.values, cv=kf, scoring="neg_root_mean_squared_error")
OUT["regression_cv"] = {
    "model": "RidgeCV (alpha tuned inside each training fold), 5-fold",
    "in_sample_r2_model1": OUT["model1_content_only"]["r2"],
    "cv_r2_mean": round(float(r2cv.mean()), 3), "cv_r2_sd": round(float(r2cv.std()), 3),
    "cv_r2_folds": [round(float(v), 3) for v in r2cv],
    "cv_rmse_mean_log10_units": round(float(rmsecv.mean()), 3),
    "cv_rmse_sd": round(float(rmsecv.std()), 3),
    "baseline_rmse_mean_predictor": round(float(df.log_views.std()), 3),
}

# ----------------------------------------------------------------------------
# 9. RELIABILITY placeholder recomputation note + dataset fingerprint
# ----------------------------------------------------------------------------
OUT["provenance"] = {
    "scraper_actor": "apidojo/tiktok-scraper (Apify actor 5K30i8aFccKNF5ICs)",
    "scraper_params": {"dateRange": "THIS_WEEK", "location": "US",
                       "maxItems_per_keyword": 5, "top_n_retained_per_run": 15,
                       "schedule": "every 7 days"},
    "ai_model": "gemini-1.5-pro (generativelanguage.googleapis.com/v1beta)",
    "ai_input": "video file uploaded via Gemini Files API (visual + audio track); "
                "captions/hashtags/metadata NOT supplied to the model",
    "ai_response_schema": ["analysis", "category", "hook", "visualHook", "style", "CTA", "tone"],
    "orchestration": "n8n workflow 'Content Analyzer (Tiktok)'; Airtable base as store",
    "dataset_sha256": hashlib.sha256(
        pd.util.hash_pandas_object(df[["reelId", "videoViewCount", "likesCount", "commentsCount"]],
                                   index=False).values.tobytes()).hexdigest()[:16],
}

df.to_csv("analytic_dataset_derived.csv", index=False)
with open("results_revision.json", "w") as f:
    json.dump(OUT, f, indent=2, default=str)

# ---------------------------------------------------------------- print
def show(d, ind=0):
    for k, v in d.items():
        if isinstance(v, dict):
            print(" " * ind + f"[{k}]"); show(v, ind + 2)
        else:
            print(" " * ind + f"{k}: {v}")
show(OUT)
