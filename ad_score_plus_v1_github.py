import os
import re
import math
import numpy as np
import pandas as pd
import librosa
from openai import OpenAI
from IPython.display import display
from typing import Optional

# ======================================================================
# 0) Setup and Transcription
# ======================================================================

# Initialize OpenAI client (Assumes API key is set as environment variable)
client = OpenAI()

# --- Define the single audio path ---
audio_path = 'voice/kinabot-Normal.m4a' 

def transcribe_audio(file_path: str, language: str = "ja") -> str:
    """Transcribe an audio file using OpenAI's gpt-4o-transcribe model."""
    if not os.path.exists(file_path):
        print(f"Warning: Audio file not found at {file_path}. Skipping transcription.")
        return ""
    try:
        if not os.getenv("OPENAI_API_KEY"):
            print("Error: OPENAI_API_KEY environment variable is not set. Cannot transcribe.")
            return ""

        with open(file_path, "rb") as f:
            tr = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=f,
                language=language
            )
        return tr.text
    except Exception as e:
        print(f"Error during transcription of {file_path}: {e}")
        return ""

# --- Perform Transcription for the single file ---
print(f"Transcribing: {audio_path}")
text_jp_hc = transcribe_audio(audio_path, language="ja")

if not text_jp_hc:
    print("WARNING: Transcription failed or returned empty text. Using a placeholder for text analysis.")
    text_jp_hc = "プロのスピーカーによるニュース原稿の読み上げです。文の構成は明確で、文法的に正しく、語彙の多様性も高い傾向があります。専門用語の使用は最小限に抑えられ、句読点も適切に利用されています。"
    
# ======================================================================
# 1) Text Processing Utilities
# ======================================================================
SENT_SPLIT_RE = re.compile(r"[。！？!?．｡\?\!]+\s*|\n+")
TOKEN_RE_ALL  = re.compile(
    r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]+"  # Kanji + Hiragana + Katakana
    r"|[A-Za-z]+"                                  # English words
    r"|\d+(?:\.\d+)?"                              # numbers
)

def split_sentences(text: str) -> list[str]:
    """Splits Japanese text into sentences."""
    text = (text or "").strip()
    return [s.strip() for s in SENT_SPLIT_RE.split(text) if s.strip()]

def tokenize(text: str) -> list[str]:
    """Tokenizes Japanese text based on defined patterns (Kanji, Kana, English, Numbers)."""
    return TOKEN_RE_ALL.findall(text or "")

# ======================================================================
# 2) Japanese Lexicons
# ======================================================================
JP_FUNCTION_WORDS = set("""
は が を に で へ と も から まで より の そして また しかし など だ です ます ない たい られる れる だろう
""".split())

DEICTIC_JP = set("これ それ あれ どれ ここ そこ あそこ どこ この その あの こう そう ああ こういう そういう ああいう".split())
UNCERTAINTY_JP = set("たぶん かもしれない でしょう らしい ようだ みたいだ おそらく".split())
FILLERS_JP = set("えー あのー えっと まあ なんか あれ うーん".split())
# AMBIG_TERMS_JP and CONNECTIVES_JP removed as features depending on them were removed

# ======================================================================
# 3) Text Feature Helpers (Lexical Diversity & Complexity) - ONLY KEPT FUNCTIONS
# ======================================================================

def freq_dist(tokens: list[str]) -> dict[str, int]:
    """Computes token frequency distribution."""
    fd = {}
    for t in tokens:
        fd[t] = fd.get(t, 0) + 1
    return fd

def yules_k(tokens: list[str]) -> float:
    """Calculates Yule's K measure of lexical diversity."""
    N = len(tokens)
    if N == 0: return 0.0
    fd = freq_dist(tokens)
    spectrum = {}
    for f in fd.values():
        spectrum[f] = spectrum.get(f, 0) + 1
    sum_i2Vi = sum((i**2) * v for i, v in spectrum.items())
    K = 10000.0 * (sum_i2Vi - N) / (N**2)
    return max(K, 0.0)

def mtld(tokens: list[str], ttr_threshold: float = 0.72) -> float:
    """Calculates Measure of Textual Lexical Diversity (MTLD)."""
    def _mtld_dir(seq):
        factors = 0
        types = set()
        token_count = 0
        for tok in seq:
            token_count += 1
            types.add(tok)
            ttr = len(types) / token_count
            if ttr <= ttr_threshold:
                factors += 1
                types = set()
                token_count = 0
        if token_count > 0:
            denom = (1 - ttr_threshold)
            factors += (1 - ttr) / denom if denom != 0 else 0
        return len(seq) / factors if factors != 0 else float('inf')
    
    if len(tokens) == 0: return 0.0
    mtld_score = (_mtld_dir(tokens) + _mtld_dir(list(reversed(tokens)))) / 2.0
    return mtld_score if mtld_score != float('inf') else 0.0

def repetition_rate(tokens: list[str]) -> float:
    """Calculates the rate of consecutive token repetitions (token[i] == token[i-1])."""
    if not tokens: return 0.0
    reps = 0
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i-1]:
            reps += 1
    return reps

# ======================================================================
# 4) Text Feature Extraction - ONLY KEPT FEATURES
# ======================================================================

def compute_text_features(text: str) -> dict:
    """Extracts a reduced set of 7 lexical and complexity features from Japanese text."""
    sents = split_sentences(text)
    toks  = tokenize(text)
    N = len(toks)
    V  = len(freq_dist(toks))

    # --- Richness (3 features kept) ---
    TTR = V / N if N else 0.0
    MTLD  = mtld(toks)
    K     = yules_k(toks)

    # --- Sentence Complexity (1 feature kept) ---
    sent_lens = [len(tokenize(s)) for s in sents] or [0]
    mean_sent = sum(sent_lens) / len(sent_lens) if len(sent_lens) else 0.0

    # --- Disfluency words (4 features kept) ---
    filler_count = sum(1 for t in toks if t in FILLERS_JP)
    deictic_count = sum(1 for t in toks if t in DEICTIC_JP)
    uncertainty_count = sum(1 for t in toks if t in UNCERTAINTY_JP)
    
    # --- Repetitions (1 feature kept) ---
    consec_rep = repetition_rate(toks)

    # Normalize helper: frequency per 100 tokens
    per100 = (lambda x: (x / N * 100.0) if N else 0.0)

    feats = {
        "tokens": N, "types": V, "sentences": len(sents),
        "ttr": TTR, "mtld": MTLD, "yule_k": K,
        "mean_sentence_len": mean_sent, 
        "fillers_per100": per100(filler_count),
        "deictic_per100": per100(deictic_count),
        "uncertainty_per100": per100(uncertainty_count),
        "consecutive_repeats_per100": per100(consec_rep),
    }
    return feats

# ======================================================================
# 5) Text Scoring (0–100): HC > AD - MODIFIED TO USE 7 FEATURES
# ======================================================================

def clamp01(x: float) -> float:
    """Clamps a value to the range [0.0, 1.0]."""
    return max(0.0, min(1.0, x))

def linear_score(x: float, lo: float, hi: float) -> float:
    """Linearly maps a value from [lo, hi] to [0, 1]. Higher is better."""
    if hi == lo: return 0.5
    return clamp01((x - lo) / (hi - lo))

def linear_score_inv(x: float, lo: float, hi: float) -> float:
    """Linearly maps a value from [lo, hi] to [1, 0]. Lower is better."""
    return 1.0 - linear_score(x, lo, hi)

def composite_text_score(feats: dict) -> tuple[float, dict]:
    """
    Calculates a composite text quality score (0-100) based on 7 features.
    """
    # --- Lexical richness (3 features) ---
    s_ttr   = linear_score(feats["ttr"],            0.20, 0.80)
    s_mtld  = linear_score(feats["mtld"],           10.0, 150.0)
    s_k     = linear_score_inv(feats["yule_k"],     50.0, 300.0)
    
    # --- Sentence complexity (1 feature) ---
    s_msl   = linear_score(feats["mean_sentence_len"], 8.0, 30.0)

    # --- Clarity / Disfluency (4 features) ---
    s_fill  = linear_score_inv(feats["fillers_per100"],        0.0, 10.0)
    s_deix  = linear_score_inv(feats["deictic_per100"],        0.0, 15.0)
    s_repc  = linear_score_inv(feats["consecutive_repeats_per100"], 0.0, 5.0)
    s_unc   = linear_score_inv(feats["uncertainty_per100"],    0.0, 10.0)
    
    # Sub-scores (Averaged using feature count)
    s_lex = (s_ttr + s_mtld + s_k) / 3.0       
    s_complex = s_msl / 1.0                    
    s_clarity = (s_fill + s_deix + s_repc + s_unc) / 4.0 
    
    # Weights (normalized sum=1.0) 
    w_lex = 0.45
    w_cmp = 0.25
    w_clr = 0.30

    # Composite Score
    total = w_lex * s_lex + w_cmp * s_complex + w_clr * s_clarity

    return round(total * 100.0, 2), {
        "s_lex": round(s_lex*100, 2),
        "s_complex": round(s_complex*100, 2),
        "s_clarity": round(s_clarity*100, 2),
    }

# ======================================================================
# 6) Text Analysis Pipeline (No change needed)
# ======================================================================

def analyze_texts(samples: list[dict]) -> pd.DataFrame:
    """
    Batch processes a list of text samples to compute features and scores.
    """
    rows = []
    for s in samples:
        text = s.get("text", "")
        if not text:
            print(f"Warning: Skipping sample with ID '{s.get('id', 'N/A')}' due to empty text.")
            continue
            
        feats = compute_text_features(text)
        score, subs = composite_text_score(feats)
        row = {
            "ID": s.get("id", ""),
            "Label": s.get("label", ""),
            "Score(0-100)": score,
            **subs,
            **feats
        }
        rows.append(row)
        
    df = pd.DataFrame(rows).sort_values(by="Score(0-100)", ascending=False).reset_index(drop=True)
    return df

# ======================================================================
# 7) Speech Feature Extraction Helpers - ONLY KEPT FEATURES
# ======================================================================

SR = 16000

def pause_features(y: np.ndarray, sr: int = SR, frame_length: int = 1024, 
                   hop_length: int = 256, rms_thresh: float = 0.03) -> dict:
    """Extracts features related to speaking activity (Segments per Minute)."""
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length).squeeze()
    speech = rms > rms_thresh
    
    # Calculate speech segments
    ch = np.diff(speech.astype(int), prepend=0, append=0)
    s_idx = np.where(ch == 1)[0]
    e_idx = np.where(ch == -1)[0]
    total = len(y) / sr

    return {
        "duration_sec": total,
        "segments_per_min": (len(s_idx) / total * 60) if total > 0 else 0.0,
    }

def f0_features(y: np.ndarray, sr: int = SR, fmin: int = 50, fmax: int = 400, 
                frame_length: int = 2048, hop_length: int = 256) -> dict:
    """Extracts fundamental frequency (F0/pitch) features (Mean, Std)."""
    try:
        f0 = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr,
                         frame_length=frame_length, hop_length=hop_length)
        f0 = f0[np.isfinite(f0)] 
    except Exception:
        f0 = np.array([])
        
    return {
        "f0_mean": float(np.mean(f0)) if f0.size else 0.0,
        "f0_std":  float(np.std(f0))  if f0.size else 0.0,
    }

def spectral_features(y: np.ndarray, sr: int = SR, hop_length: int = 256) -> dict:
    """Extracts spectral features (Centroid, RMS, Contrast)."""
    n_fft = 1024
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) + 1e-10

    centroid = librosa.feature.spectral_centroid(S=S, sr=sr).squeeze()
    rms = librosa.feature.rms(S=S, frame_length=n_fft).squeeze() 

    return {
        "spec_centroid_mean": float(np.mean(centroid)),
        "spec_centroid_std":  float(np.std(centroid)), # Kept for completeness, not used in score
        "rms_mean": float(np.mean(rms)),
    }

def mfcc_features(y: np.ndarray, sr: int = SR, n_mfcc: int = 20, hop_length: int = 256) -> dict:
    """Extracts Mel-Frequency Cepstral Coefficients (MFCCs) and their deltas (mean and std)."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    dm   = librosa.feature.delta(mfcc)
    ddm = librosa.feature.delta(mfcc, order=2)
    
    feats = {}
    for name, M in [("mfcc", mfcc), ("d_mfcc", dm), ("dd_mfcc", ddm)]:
        mean = M.mean(axis=1)
        std = M.std(axis=1)
        for i in range(M.shape[0]):
            feats[f"{name}{i+1}_mean"] = float(mean[i])
            feats[f"{name}{i+1}_std"] = float(std[i])
    return feats

def extract_speech_features(path: str, sr: int = SR) -> Optional[dict]:
    """Loads audio and extracts all speech features."""
    try:
        y, sr_actual = librosa.load(path, sr=sr, mono=True)
        if sr_actual != sr:
            print(f"Warning: Actual sample rate {sr_actual} differs from target {sr}.")
        
        y = librosa.util.normalize(y)
        
        feats = {}
        feats.update(mfcc_features(y, sr))
        feats.update(spectral_features(y, sr))
        feats.update(pause_features(y, sr))
        feats.update(f0_features(y, sr))
        return feats
    except Exception as e:
        print(f"Error loading or processing audio file {path}: {e}")
        return None

# ======================================================================
# 8) Speech Rule-Based Scoring - MODIFIED TO USE 4 FEATURES
# ======================================================================

def clip_scale(x: float, lo: float, hi: float) -> float:
    """Linearly scales a value from [lo, hi] to [0, 1] and clips it."""
    if hi == lo: return 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))

def score_from_speech_feats(f: dict) -> tuple[int, float, float, float]:
    """
    Calculates a rule-based speech quality score (1-100) based on 4 features.
    """
    # Total features used: 4 (segments_per_min, f0_std, rms_mean, spec_centroid_mean)

    # 1) Speaking Activity (SA) - 1 feature
    SA = clip_scale(f['segments_per_min'], 0.0, 120.0) 

    # 2) Prosody Variation (PV) - 1 feature
    PV = clip_scale(f['f0_std'], 0.0, 60.0)

    # 3) Energy/Voice Quality (EN) - 2 features
    en_parts = [
        clip_scale(f['rms_mean'], 0.005, 0.050),            
        clip_scale(f['spec_centroid_mean'], 800.0, 3500.0), 
    ]
    EN = float(np.mean(en_parts)) # Implicitly divides by 2.0

    # Weighted blend 
    score01 = 0.5 * SA + 0.3 * PV + 0.2 * EN
    # Map [0,1] to [1, 100]
    score = int(np.clip(np.round(1 + 99 * score01), 1, 100))
    return score, SA, PV, EN

def evaluate_speech(files: list[tuple[str, str]]) -> pd.DataFrame:
    """
    Batch processes a list of audio files to compute speech features and scores.
    """
    rows = []
    for path, tag in files:
        if not os.path.exists(path):
            print(f"Error: File not found: {path}. Skipping.")
            continue
            
        f = extract_speech_features(path)
        if f is None:
            print(f"Skipping scoring for {path} due to feature extraction error.")
            continue

        score, SA, PV, EN = score_from_speech_feats(f)
        
        # Select and round key features for the output DataFrame 
        rows.append({
            "file": os.path.basename(path),
            "tag": tag,
            "score_1_100": score,
            "SA_activity": round(SA, 3),
            "PV_prosody": round(PV, 3),
            "EN_energy":  round(EN, 3),
            "segments_per_min": round(f["segments_per_min"], 1),
            "f0_std": round(f["f0_std"], 2),
            "spec_centroid_mean": round(f["spec_centroid_mean"], 1),
            "rms_mean": round(f["rms_mean"], 4),
        })
        
    df = pd.DataFrame(rows)
    df["prediction"] = np.where(df["score_1_100"] >= 70, "Non-AD (Healthier)", "AD Tendency (Needs attention)")
    
    return df

# ======================================================================
# 9) Text and Speech Score Combination (No change needed)
# ======================================================================

def combine_text_speech(
    text_df: pd.DataFrame,
    speech_df: pd.DataFrame,
    w_text: float = 0.5,
    w_speech: float = 0.5,
    join_on: Optional[str] = None,
    output_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Combines text and speech analysis results into a single DataFrame 
    with a weighted final score.
    """
    # ---- Sanity Checks and Weight Normalization ----
    if "Score(0-100)" not in text_df.columns or "score_1_100" not in speech_df.columns:
        raise ValueError('Both DataFrames must contain their respective score columns ("Score(0-100)" and "score_1_100").')

    w_sum = w_text + w_speech
    if w_sum <= 0: w_text, w_speech, w_sum = 0.5, 0.5, 1.0
    w_text /= w_sum
    w_speech /= w_sum

    # Select/prepare score columns (ensure they are numeric and clipped)
    t_score = pd.to_numeric(text_df["Score(0-100)"], errors="coerce").fillna(0.0).clip(0, 100)
    s_score = pd.to_numeric(speech_df["score_1_100"], errors="coerce").fillna(0.0).clip(0, 100)

    # --- Prepare DataFrames for merging ---
    t_keep_cols = [c for c in text_df.columns if c != "Score(0-100)"]
    s_keep_cols = [c for c in speech_df.columns if c != "score_1_100"]

    def prefix_cols_map(cols: list[str], prefix: str, keep: set) -> dict:
        """Helper to create a column rename map, keeping 'keep' columns unprefixed."""
        return {c: (c if c in keep else f"{prefix}{c}") for c in cols}

    text_keep_for_prefix = set([join_on]) if join_on and join_on in text_df.columns else set()
    speech_keep_for_prefix = set([join_on]) if join_on and join_on in speech_df.columns else set()

    t_pref = text_df[t_keep_cols].rename(columns=prefix_cols_map(t_keep_cols, "TEXT_", text_keep_for_prefix))
    s_pref = speech_df[s_keep_cols].rename(columns=prefix_cols_map(s_keep_cols, "SPEECH_", speech_keep_for_prefix))

    # Add the individual score columns, prefixed (for transparency)
    t_pref["TEXT_Score"] = t_score.values
    s_pref["SPEECH_score"] = s_score.values

    # ---- Merge / Align ----
    if join_on and (join_on in t_pref.columns) and (join_on in s_pref.columns):
        merged = pd.merge(t_pref, s_pref, on=join_on, how="inner", validate="m:1")
        merged_scores = pd.merge(
            text_df[[join_on]].assign(t_score=t_score.values),
            speech_df[[c for c in speech_df.columns if c in [join_on, "score_1_100"]]],
            on=join_on, how="inner", validate="m:1"
        )
        aligned_combined = (w_text * merged_scores["t_score"] + w_speech * merged_scores["score_1_100"]).clip(0, 100)
        
        merged.insert(1 if join_on in merged.columns else 0, "Combined_Score(0-100)", aligned_combined.values)
    else:
        n = min(len(t_pref), len(s_pref))
        if len(t_pref) != len(s_pref):
            print(f"[Warning] Row counts differ (text={len(t_pref)}, speech={len(s_pref)}). "
                  f"Truncating to {n} by row order.")
        merged = pd.concat([t_pref.iloc[:n].reset_index(drop=True),
                            s_pref.iloc[:n].reset_index(drop=True)], axis=1)
        
        combined = (w_text * t_score.iloc[:n] + w_speech * s_score.iloc[:n]).clip(0, 100)
        merged.insert(0, "Combined_Score(0-100)", combined.values)

    # Save if requested
    if output_csv:
        merged.to_csv(output_csv, index=False, encoding="utf-8-sig")

    return merged

# ======================================================================
# 10) Execution / Example
# ======================================================================

# Define audio file paths for the speech analysis 
FILES = [
    (audio_path, "Prof"),
]

# Define transcription results for the text analysis
SAMPLES = [
    {"id": "JP_Normal", "label": "Broadcaster", "text": text_jp_hc},
]


print ("\n\n=========================\nStarting Text Analysis\n=========================\n\n")
# --- TEXT ANALYSIS ---
df_text = analyze_texts(SAMPLES) 

# Display compact text features (Uses the 7 features kept for scoring)
cols_compact = ["ID", "Label", "Score(0-100)", "s_lex", "s_complex", "s_clarity",
                "ttr", "mtld", # Lexical Richness 
                "mean_sentence_len", # Complexity
                "deictic_per100", "fillers_per100", "uncertainty_per100",
                "consecutive_repeats_per100"] # Clarity
display(df_text[cols_compact])


print ("\n\n=========================\nStarting Speech Analysis\n=========================\n\n")
# --- SPEECH ANALYSIS ---
df_speech = evaluate_speech(FILES) 

if df_speech.empty:
    print("WARNING: Speech analysis failed (file not found or error). Using DUMMY data for combination.")
    # Dummy data structure only includes the 4 features still used for scoring
    dummy_speech_data = [
        {"file": os.path.basename(audio_path), "tag": "Prof", "score_1_100": 95, "SA_activity": 0.8, "PV_prosody": 0.9, "EN_energy": 0.9,
         "segments_per_min": 100.0, "f0_std": 50.0, "spec_centroid_mean": 2000.0, "rms_mean": 0.03},
    ]
    df_speech = pd.DataFrame(dummy_speech_data) 
    df_speech["prediction"] = np.where(df_speech["score_1_100"] >= 70, "Non-AD (Healthier)", "AD Tendency (Needs attention)")

# Display speech results
print(df_speech[["file", "tag", "score_1_100", "SA_activity", "PV_prosody", "EN_energy", "prediction"]].to_string(index=False))
print("\nRule-based speech scores saved to rule_based_scores.csv (or using dummy if file failed)")


print ("\n\n=========================\nCombining Text + Speech\n=========================\n\n")
# --- COMBINE SCORES ---
combo = combine_text_speech(
    df_text,
    df_speech,
    w_text=0.05,                 # Custom weights 
    w_speech=0.95,
    join_on=None,               # Align by row index
    output_csv="overall_scores.csv"
)

# Display combined results
display(combo)
print("\nCombined scores saved to overall_scores.csv")
