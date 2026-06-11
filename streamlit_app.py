"""
Streamlit web application for recording audio and analyzing with Kina.
Enhanced with timer and visual recording feedback.
"""
import streamlit as st
import speech_recognition as sr
from pydub import AudioSegment
from textblob import TextBlob
import textwrap
import re
import os
from datetime import datetime
from audio_recorder_streamlit import audio_recorder

# Page config
st.set_page_config(
    page_title="Kina - Speech to Cognitive Insights",
    page_icon="🎤",
    layout="wide"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.05); opacity: 0.8; }
    }
    
    .recording-status {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
        font-size: 20px;
        font-weight: bold;
    }
    
    .recording-active {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        animation: pulse 1.5s infinite;
    }
    
    .recording-ready {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .timer-display {
        font-size: 56px;
        font-weight: bold;
        text-align: center;
        color: #ef4444;
        padding: 30px;
        background: #fee2e2;
        border-radius: 15px;
        margin: 20px 0;
        font-family: 'Courier New', monospace;
    }
    
    .timer-complete {
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        color: #667eea;
        padding: 25px;
        background: #e0e7ff;
        border-radius: 15px;
        margin: 20px 0;
        font-family: 'Courier New', monospace;
    }
    
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    
    .stButton > button {
        transition: all 0.3s ease;
    }
    
    .recording-dot {
        display: inline-block;
        width: 16px;
        height: 16px;
        background: #ef4444;
        border-radius: 50%;
        margin-right: 10px;
        animation: pulse 1s infinite;
    }
</style>
""", unsafe_allow_html=True)

# Ensure recordings directory exists
os.makedirs('recordings', exist_ok=True)

def format_text(text, width=80):
    sentences = text.replace('. ', '.\n').replace('? ', '?\n').replace('! ', '!\n')
    return textwrap.fill(sentences, width=width)

def lexical_diversity(text):
    words = re.findall(r'\b\w+\b', text.lower())
    total_words = len(words)
    unique_words = len(set(words))
    score = unique_words / total_words if total_words else 0
    return score, total_words, unique_words

def sentence_complexity(text):
    sentences = re.split(r'[.!?]', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    num_sentences = len(sentences)
    words = re.findall(r'\b\w+\b', text)
    
    avg_sentence_length = len(words) / num_sentences if num_sentences else 0
    
    conjunctions = ['and', 'but', 'or', 'because', 'although', 'since', 'while', 'if', 'when', 'though']
    conjunction_count = sum(text.lower().count(c) for c in conjunctions)
    
    feedback = "✅ Balanced complexity." if avg_sentence_length > 12 and conjunction_count >= num_sentences else "⚠️ Simple or flat sentence structure."
    
    return avg_sentence_length, conjunction_count, feedback

def get_audio_duration(audio_bytes):
    """Calculate actual audio duration from bytes"""
    try:
        temp_file = "temp_duration.wav"
        with open(temp_file, 'wb') as f:
            f.write(audio_bytes)
        audio = AudioSegment.from_file(temp_file)
        duration = len(audio) / 1000.0  # Convert to seconds
        os.remove(temp_file)
        return duration
    except:
        return 0

def analyze_audio(audio_bytes, language='en-US'):
    """Analyze audio and return results"""
    recognizer = sr.Recognizer()
    temp_wav_file = "temp_streamlit.wav"
    
    try:
        # Save audio bytes to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        audio_path = f'recordings/recording_{timestamp}.wav'
        
        with open(audio_path, 'wb') as f:
            f.write(audio_bytes)
        
        # Get actual duration
        audio = AudioSegment.from_file(audio_path)
        duration = len(audio) / 1000.0  # Convert to seconds
        
        # Preprocess audio
        audio = audio.set_frame_rate(16000)
        audio.export(temp_wav_file, format="wav")
        
        with sr.AudioFile(temp_wav_file) as source:
            audio_data = recognizer.record(source)
        
        # Transcription
        text = recognizer.recognize_google(audio_data, language=language)
        formatted_text = format_text(text)
        
        # Sentiment
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Lexical Diversity
        diversity, total_words, unique_words = lexical_diversity(text)
        lex_pattern = "✅ Diverse vocabulary." if diversity > 0.5 else "⚠️ Repeated words or less variety."
        
        # Sentence Complexity
        avg_len, conj_count, complexity_feedback = sentence_complexity(text)
        
        return {
            'success': True,
            'transcription': formatted_text,
            'raw_text': text,
            'duration': duration,
            'sentiment': {
                'polarity': polarity,
                'subjectivity': subjectivity
            },
            'lexical': {
                'total_words': total_words,
                'unique_words': unique_words,
                'diversity_score': diversity,
                'pattern': lex_pattern
            },
            'complexity': {
                'avg_sentence_length': avg_len,
                'conjunction_count': conj_count,
                'feedback': complexity_feedback
            },
            'audio_path': audio_path
        }
    
    except sr.UnknownValueError:
        return {'success': False, 'error': 'Speech Recognition could not understand audio.'}
    except sr.RequestError as e:
        return {'success': False, 'error': f'Could not request results from Speech Recognition service: {e}'}
    except Exception as e:
        return {'success': False, 'error': f'An error occurred: {e}'}
    finally:
        # Cleanup temp file
        if os.path.exists(temp_wav_file):
            os.remove(temp_wav_file)

# Initialize session state
if 'last_audio_bytes' not in st.session_state:
    st.session_state.last_audio_bytes = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'recording_duration' not in st.session_state:
    st.session_state.recording_duration = 0

# App UI
st.title("🎤 Kina")
st.subheader("Transforming Speech into Cognitive Insights")

st.markdown("---")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    language = st.selectbox(
        "Language",
        options=[
            ("English (US)", "en-US"),
            ("English (UK)", "en-GB"),
            ("Spanish", "es-ES"),
            ("French", "fr-FR"),
            ("German", "de-DE"),
            ("Japanese", "ja-JP"),
            ("Chinese (Simplified)", "zh-CN")
        ],
        format_func=lambda x: x[0]
    )
    language_code = language[1]
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    Kina analyzes speech patterns to detect cognitive changes through:
    - **Lexical diversity** - vocabulary richness
    - **Sentence complexity** - structure analysis
    - **Sentiment analysis** - emotional tone
    """)
    
    st.markdown("---")
    st.markdown("### How to Use")
    st.markdown("""
    1. **Click** the microphone button to start recording
    2. **Speak** clearly for 10-30 seconds
    3. **Click** the red button to stop recording
    4. **View** your cognitive analysis results
    """)

# Main content
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 🎙️ Record Audio")
    
    # Audio recorder with custom styling
    audio_bytes = audio_recorder(
        text="",
        recording_color="#ef4444",
        neutral_color="#667eea",
        icon_name="microphone",
        icon_size="4x",
        sample_rate=16000,
        key="audio_recorder"
    )
    
    # Recording status and timer
    if audio_bytes and audio_bytes != st.session_state.last_audio_bytes:
        # New recording detected
        st.session_state.last_audio_bytes = audio_bytes
        duration = get_audio_duration(audio_bytes)
        st.session_state.recording_duration = duration
        
        # Show completion status with timer
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        st.markdown(f"""
            <div class="timer-complete">
                ⏱️ {minutes:02d}:{seconds:02d}
            </div>
        """, unsafe_allow_html=True)
        
        st.success(f"✅ Recording complete! Duration: {duration:.1f}s")
        
    elif audio_bytes:
        # Same recording, show duration
        if st.session_state.recording_duration > 0:
            duration = st.session_state.recording_duration
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            st.markdown(f"""
                <div class="timer-complete">
                    ⏱️ {minutes:02d}:{seconds:02d}
                </div>
            """, unsafe_allow_html=True)
    else:
        # No recording
        st.markdown("""
            <div class="recording-status recording-ready">
                🎙️ Click the microphone to start recording
            </div>
        """, unsafe_allow_html=True)
        st.session_state.recording_duration = 0
    
    # Recording tips
    with st.expander("📋 Recording Tips"):
        st.markdown("""
        - **Environment**: Find a quiet space
        - **Distance**: Keep microphone 6-12 inches away
        - **Clarity**: Speak naturally and clearly
        - **Duration**: Aim for 10-30 seconds
        - **Content**: Describe your day or tell a story
        """)

with col2:
    st.markdown("### 📊 Analysis Results")
    
    if audio_bytes:
        # Analyze if new recording
        if audio_bytes != st.session_state.get('analyzed_audio'):
            with st.spinner("🔍 Analyzing your speech..."):
                results = analyze_audio(audio_bytes, language_code)
                st.session_state.analysis_results = results
                st.session_state.analyzed_audio = audio_bytes
        else:
            results = st.session_state.analysis_results
        
        if results and results['success']:
            st.success("✅ Analysis complete!")
            
            # Transcription
            st.markdown("#### 📝 Transcription")
            st.text_area("", results['transcription'], height=120, disabled=True, label_visibility="collapsed")
            
            # Create three columns for metrics
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.markdown("#### 😊 Sentiment")
                polarity_emoji = "😊" if results['sentiment']['polarity'] > 0.1 else "😐" if results['sentiment']['polarity'] > -0.1 else "😔"
                st.metric(f"{polarity_emoji} Polarity", f"{results['sentiment']['polarity']:.2f}", 
                         help="−1 = negative, +1 = positive")
                st.metric("Subjectivity", f"{results['sentiment']['subjectivity']:.2f}",
                         help="0 = objective, 1 = subjective")
            
            with metric_col2:
                st.markdown("#### 📚 Lexical Diversity")
                st.metric("Total Words", results['lexical']['total_words'])
                st.metric("Unique Words", results['lexical']['unique_words'])
                st.metric("Diversity Score", f"{results['lexical']['diversity_score']:.2f}",
                         help="Higher is better (0-1 scale)")
                if results['lexical']['diversity_score'] > 0.5:
                    st.success(results['lexical']['pattern'])
                else:
                    st.warning(results['lexical']['pattern'])
            
            with metric_col3:
                st.markdown("#### 🧠 Complexity")
                st.metric("Avg Sentence Length", f"{results['complexity']['avg_sentence_length']:.1f} words")
                st.metric("Conjunctions", results['complexity']['conjunction_count'])
                st.metric("Recording Duration", f"{results['duration']:.1f}s")
                if "✅" in results['complexity']['feedback']:
                    st.success(results['complexity']['feedback'])
                else:
                    st.warning(results['complexity']['feedback'])
            
            # Show saved file path
            st.markdown("---")
            st.caption(f"📁 Recording saved to: `{results['audio_path']}`")
            
        elif results:
            st.error(f"❌ {results['error']}")
            st.info("💡 Try speaking more clearly or check your microphone permissions.")
    else:
        # Show placeholder
        st.info("👆 Click the microphone button to start recording")
        st.markdown("""
        <div style="text-align: center; padding: 40px; color: #9ca3af;">
            <div style="font-size: 64px; margin-bottom: 20px;">🎤</div>
            <div style="font-size: 20px;">Your analysis results will appear here</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.caption("Kina - Detect cognitive changes through natural conversation patterns")
