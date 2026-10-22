"""
Appendix A — Hashtag Classification Dictionary and Caption Regular Expressions
==============================================================================
Reproduced verbatim from analysis_revision.py.
These definitions are used in Section 4.4 (hashtag layering features) of the
manuscript.

Reference
---------
Chansanam, W. et al. (2027). Prevalence is not advantage: A reproducible audit
of content features, K-pop salience, and the limits of predicting algorithmically
mediated visibility on TikTok. International Journal of Data and Network Science,
11. https://doi.org/10.5267/j.ijdns.2026.10.012
"""

import re

# ---------------------------------------------------------------------------
# A1. HASHTAG CLASSIFICATION DICTIONARY
# ---------------------------------------------------------------------------

# Algorithmic/discovery hashtags — tags whose primary function is platform
# amplification rather than topical description.
ALGO = {
    "fyp", "fypシ", "foryou", "foryoupage", "foryourpage",
    "viral", "viralvideo", "viralvideos",
    "trending", "trend",
    "tiktokviral", "viraltiktok",
    "xyzbca",
    "fyppppppppppppppppppppppp",
    "tiktok", "explore", "explorepage", "fy",
}

# Niche/community hashtags — tags signalling an established fan community or
# content sub-genre. Matched by substring regex.
NICHE_PAT = re.compile(
    r"(fan|fans|stan|army|blink|once|carat|moa|nctzen|bunnies|"
    r"tiny|zenz|fandom|community|tok$|lovers|nation|edit|edits|"
    r"cover|challenge|squad|club)",
    re.I,
)


def classify_tag(t: str) -> str:
    """Classify a single hashtag token into one of three tiers.

    Parameters
    ----------
    t : str
        A single normalised (lowercased, stripped) hashtag token.

    Returns
    -------
    str
        One of ``"algorithmic"``, ``"niche"``, or ``"descriptive"``.
    """
    if t in ALGO:
        return "algorithmic"
    if NICHE_PAT.search(t):
        return "niche"
    return "descriptive"


# ---------------------------------------------------------------------------
# A2. CAPTION REGULAR EXPRESSIONS
# ---------------------------------------------------------------------------

# Second-person address markers
SECOND = re.compile(r"\b(you|your|yours|y'?all|u|ur)\b", re.I)

# Intrigue / curiosity-gap hooks
CURIO = re.compile(
    r"(wait for it|you won'?t believe|guess|secret|watch till|watch until|"
    r"this is why|find out|part \d|pov|don'?t scroll|omg|\?{2,}|…|\.\.\.)",
    re.I,
)

# Problem-framing language
PROBLEM = re.compile(
    r"(problem|struggle|hate when|tired of|why does|can'?t|stop|"
    r"mistake|wrong|fix|help|hard|annoying|issue|no one)",
    re.I,
)

# Imperative call-to-action openers
IMPER = re.compile(
    r"^\s*(try|watch|check|listen|look|stream|follow|comment|tag|save|do)\b",
    re.I,
)
