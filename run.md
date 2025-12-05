# README: Spoken Content Quality Assessment Script (Reduced Feature Set)

This Python script, **`ad_score_plus_v1_github.py`**, provides a comprehensive quality assessment score for spoken Japanese content by analyzing both its **linguistic complexity** (Text Score) and **acoustic quality** (Speech Score).

---

## 💡 Overview & Features

The script processes an audio file through two main stages:

1.  **Transcription:** Uses the **OpenAI Whisper API (`gpt-4o-transcribe`)** to convert the Japanese audio into text.
2.  **Scoring:** Calculates two independent scores, which are then combined into a final weighted result (default: **5% Text Score, 95% Speech Score**).

### 📝 Text Features (Reduced Subset: 7 Features Used)

The **Text Score** evaluates the content's complexity and clarity, focusing on features that typically indicate organized and fluent linguistic output.

| Category | Key Features Used |
| :--- | :--- |
| **Lexical Richness** | TTR (Type-Token Ratio), MTLD (Measure of Textual Lexical Diversity), Yule's K (Inverse) |
| **Syntactic Complexity** | Mean Sentence Length |
| **Clarity/Disfluency** | Fillers, Deictic terms, Consecutive Repetitions, Uncertainty Words |

### 🗣️ Speech Features (Reduced Subset: 4 Features Used)

The **Speech Score** evaluates core acoustic characteristics, indicative of voice stability and speaking rhythm.

| Dimension | Key Features Used |
| :--- | :--- |
| **Speaking Activity (SA)** | Segments per minute (Speaking Rate) |
| **Prosody Variation (PV)** | F0 Standard Deviation (Pitch variation) |
| **Energy/Quality (EN)** | RMS Mean (Loudness/Energy), Spectral Centroid Mean (Clarity/Brightness) |

---

## ⚙️ Setup and Prerequisites

To run this script, you will need a Python environment with the following dependencies installed, and an OpenAI API key.

### 1. Dependencies

Install the required libraries:

```bash
pip install openai pandas numpy librosa
```

For robust audio file loading (especially .m4a files), you should also ensure you have FFmpeg installed on your system.


### 2. How to Run the Script
Save the Code: Save the provided Python code as a file named `ad_score_plus_v1_github.py`.

Set Audio Path: Ensure the audio file you want to analyze is accessible, and update the audio_path variable near the top of the script to point to your specific file if it's not in the default location:

```bash
# Example: If your audio is in the current directory
audio_path = 'my_audio_file.m4a' 
```

Execute: Run the script from your terminal:


`python ad_score_plus_v1_github.py`

### 3. Output

The script will print the results to the console and generate a CSV file in the execution directory:


`combined_scores.csv`: The final combined score and the sub-scores from both the text and speech analysis.
