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
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from io import BytesIO

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

    conjunctions = ['and', 'but', 'or', 'because', 'although', 'since',
                    'while', 'if', 'when', 'though']
    conjunction_count = sum(text.lower().count(c) for c in conjunctions)

    if avg_sentence_length > 12 and conjunction_count >= num_sentences:
        feedback = "✅ Balanced complexity."
    else:
        feedback = "⚠️ Simple or flat sentence structure."

    return avg_sentence_length, conjunction_count, feedback


def calculate_cognitive_score(results):
    """
    Calculate overall cognitive health score and estimated cognitive age.

    Returns:
        dict: Contains overall_score (0-100), cognitive_age, risk_level, and recommendations
    """

    # Extract metrics
    diversity_score = results['lexical']['diversity_score']
    total_words = results['lexical']['total_words']
    avg_sentence_length = results['complexity']['avg_sentence_length']
    conjunction_count = results['complexity']['conjunction_count']
    duration = results['duration']
    polarity = results['sentiment']['polarity']

    # Calculate component scores (0-100 scale)

    # 1. Lexical Diversity Score (30% weight)
    # Healthy range: 0.6-0.9, Optimal: 0.75+
    if diversity_score >= 0.75:
        diversity_component = 100
    elif diversity_score >= 0.6:
        diversity_component = 60 + (diversity_score - 0.6) * 266.67  # Scale 0.6-0.75 to 60-100
    elif diversity_score >= 0.4:
        diversity_component = 30 + (diversity_score - 0.4) * 150     # Scale 0.4-0.6 to 30-60
    else:
        diversity_component = max(0, diversity_score * 75)           # Scale 0-0.4 to 0-30

    # 2. Speech Fluency Score (25% weight)
    # Based on words per second (optimal: 2-3 words/second)
    words_per_second = total_words / duration if duration > 0 else 0
    if 2.0 <= words_per_second <= 3.0:
        fluency_component = 100
    elif 1.5 <= words_per_second < 2.0 or 3.0 < words_per_second <= 3.5:
        fluency_component = 80
    elif 1.0 <= words_per_second < 1.5 or 3.5 < words_per_second <= 4.0:
        fluency_component = 60
    elif 0.5 <= words_per_second < 1.0 or 4.0 < words_per_second <= 5.0:
        fluency_component = 40
    else:
        fluency_component = 20

    # 3. Sentence Complexity Score (25% weight)
    # Optimal sentence length: 12-20 words, with good conjunction usage
    if 12 <= avg_sentence_length <= 20 and conjunction_count >= 1:
        complexity_component = 100
    elif 8 <= avg_sentence_length < 12 or 20 < avg_sentence_length <= 25:
        complexity_component = 80
    elif 6 <= avg_sentence_length < 8 or 25 < avg_sentence_length <= 30:
        complexity_component = 60
    elif 4 <= avg_sentence_length < 6 or avg_sentence_length > 30:
        complexity_component = 40
    else:
        complexity_component = 20

    # 4. Emotional Expression Score (20% weight)
    # Healthy emotional range: slight positive to neutral
    if -0.1 <= polarity <= 0.3:
        emotion_component = 100
    elif -0.3 <= polarity < -0.1 or 0.3 < polarity <= 0.5:
        emotion_component = 80
    elif -0.5 <= polarity < -0.3 or 0.5 < polarity <= 0.7:
        emotion_component = 60
    else:
        emotion_component = 40

    # Calculate weighted overall score
    overall_score = (
        diversity_component * 0.30 +
        fluency_component * 0.25 +
        complexity_component * 0.25 +
        emotion_component * 0.20
    )

    # Determine risk level
    if overall_score >= 80:
        risk_level = "Low Risk"
        risk_color = "🟢"
        risk_description = "Excellent cognitive health indicators"
    elif overall_score >= 65:
        risk_level = "Low-Moderate Risk"
        risk_color = "🟡"
        risk_description = "Good cognitive health with minor areas for attention"
    elif overall_score >= 50:
        risk_level = "Moderate Risk"
        risk_color = "🟠"
        risk_description = "Some cognitive health concerns detected"
    else:
        risk_level = "Higher Risk"
        risk_color = "🔴"
        risk_description = "Multiple cognitive health indicators suggest consultation recommended"

    # Estimate cognitive age (simplified model)
    # Base age adjustment on score deviation from optimal (85)
    score_deviation = 85 - overall_score
    age_adjustment = score_deviation * 0.3  # Each point below 85 adds ~0.3 years

    # Assume baseline cognitive age of 35 for healthy adult
    baseline_age = 35
    estimated_cognitive_age = max(20, baseline_age + age_adjustment)

    # Generate recommendations
    recommendations = []

    if diversity_component < 70:
        recommendations.append("📚 Expand vocabulary through reading diverse materials")

    if fluency_component < 70:
        if words_per_second < 1.5:
            recommendations.append("🗣️ Practice speaking exercises to improve fluency")
        else:
            recommendations.append("⏸️ Practice paced speaking to improve clarity")

    if complexity_component < 70:
        recommendations.append("🧠 Practice using varied sentence structures and connecting words")

    if emotion_component < 70:
        recommendations.append("😊 Consider activities that promote emotional well-being")

    if not recommendations:
        recommendations.append("🎉 Excellent results! Continue maintaining "
                               "cognitive health through regular mental activities")

    return {
        'overall_score': round(overall_score, 1),
        'cognitive_age': round(estimated_cognitive_age, 1),
        'risk_level': risk_level,
        'risk_color': risk_color,
        'risk_description': risk_description,
        'component_scores': {
            'lexical_diversity': round(diversity_component, 1),
            'speech_fluency': round(fluency_component, 1),
            'sentence_complexity': round(complexity_component, 1),
            'emotional_expression': round(emotion_component, 1)
        },
        'recommendations': recommendations,
        'words_per_second': round(words_per_second, 2)
    }


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
    except Exception:
        return 0


def generate_pdf_report(results, language_name):
    """Generate a PDF report of the analysis results"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5 * inch)

    # Calculate cognitive score
    cognitive_results = calculate_cognitive_score(results)

    # Get styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        textColor=colors.HexColor('#667eea'),
        alignment=1  # Center alignment
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        textColor=colors.HexColor('#374151'),
        borderWidth=1,
        borderColor=colors.HexColor('#e5e7eb'),
        borderPadding=8,
        backColor=colors.HexColor('#f9fafb')
    )

    # Build the document
    story = []

    # Title
    story.append(Paragraph("🎤 KINA Speech Analysis Report", title_style))
    story.append(Spacer(1, 20))

    # Header info table
    header_data = [
        ['Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ['Recording Duration:', f"{results['duration']:.1f} seconds"],
        ['Language:', language_name],
        ['Audio File:', results['audio_path']]
    ]

    header_table = Table(header_data, colWidths=[2 * inch, 4 * inch])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f3f4f6')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e5e7eb')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))

    story.append(header_table)
    story.append(Spacer(1, 20))

    # Cognitive Health Assessment
    story.append(Paragraph("🎯 COGNITIVE HEALTH ASSESSMENT", heading_style))
    story.append(Spacer(1, 10))

    cognitive_data = [
        ['Overall Cognitive Score:', f"{cognitive_results['overall_score']}/100"],
        ['Estimated Cognitive Age:', f"{cognitive_results['cognitive_age']} years"],
        ['Risk Level:', f"{cognitive_results['risk_level']}"],
        ['Assessment:', cognitive_results['risk_description']]
    ]

    cognitive_table = Table(cognitive_data, colWidths=[2 * inch, 4 * inch])
    cognitive_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e0e7ff')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e5e7eb')),
    ]))

    story.append(cognitive_table)
    story.append(Spacer(1, 20))

    # Component Scores
    story.append(Paragraph("� DETAILED SCORE BREAKDOWN", heading_style))
    story.append(Spacer(1, 10))

    component_data = [
        ['Lexical Diversity:',
         f"{cognitive_results['component_scores']['lexical_diversity']:.1f}/100"],
        ['Speech Fluency:',
         f"{cognitive_results['component_scores']['speech_fluency']:.1f}/100"],
        ['Sentence Complexity:',
         f"{cognitive_results['component_scores']['sentence_complexity']:.1f}/100"],
        ['Emotional Expression:',
         f"{cognitive_results['component_scores']['emotional_expression']:.1f}/100"],
        ['Speaking Rate:', f"{cognitive_results['words_per_second']} words/second"]
    ]

    component_table = Table(component_data, colWidths=[2 * inch, 4 * inch])
    component_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f9ff')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e5e7eb')),
    ]))

    story.append(component_table)
    story.append(Spacer(1, 20))

    # Transcription
    story.append(Paragraph("📝 TRANSCRIPTION", heading_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph(results['transcription'], styles['Normal']))
    story.append(Spacer(1, 20))

    # Sentiment Analysis
    story.append(Paragraph("😊 SENTIMENT ANALYSIS", heading_style))
    story.append(Spacer(1, 10))

    sentiment_data = [
        ['Polarity Score:', f"{results['sentiment']['polarity']:.2f}",
         '(−1 = negative, +1 = positive)'],
        ['Subjectivity Score:', f"{results['sentiment']['subjectivity']:.2f}",
         '(0 = objective, 1 = subjective)']
    ]

    sentiment_table = Table(sentiment_data, colWidths=[2 * inch, 1 * inch, 3 * inch])
    sentiment_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#fef3c7')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e5e7eb')),
    ]))

    story.append(sentiment_table)
    story.append(Spacer(1, 20))

    # Lexical Diversity
    story.append(Paragraph("📚 LEXICAL DIVERSITY", heading_style))
    story.append(Spacer(1, 10))

    lexical_data = [
        ['Total Words:', str(results['lexical']['total_words'])],
        ['Unique Words:', str(results['lexical']['unique_words'])],
        ['Diversity Score:', f"{results['lexical']['diversity_score']:.2f} (Unique / Total)"],
        ['Pattern Assessment:', results['lexical']['pattern']]
    ]

    lexical_table = Table(lexical_data, colWidths=[2 * inch, 4 * inch])
    lexical_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#dbeafe')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e5e7eb')),
    ]))

    story.append(lexical_table)
    story.append(Spacer(1, 20))

    # Sentence Complexity
    story.append(Paragraph("🧠 SENTENCE COMPLEXITY", heading_style))
    story.append(Spacer(1, 10))

    complexity_data = [
        ['Average Sentence Length:', f"{results['complexity']['avg_sentence_length']:.1f} words"],
        ['Conjunction Count:', str(results['complexity']['conjunction_count'])],
        ['Complexity Feedback:', results['complexity']['feedback']]
    ]

    complexity_table = Table(complexity_data, colWidths=[2 * inch, 4 * inch])
    complexity_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f3e8ff')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e5e7eb')),
    ]))

    story.append(complexity_table)
    story.append(Spacer(1, 20))

    # Recommendations
    if cognitive_results['recommendations']:
        story.append(Paragraph("💡 PERSONALIZED RECOMMENDATIONS", heading_style))
        story.append(Spacer(1, 10))

        for i, rec in enumerate(cognitive_results['recommendations'], 1):
            story.append(Paragraph(f"{i}. {rec}", styles['Normal']))
            story.append(Spacer(1, 5))

        story.append(Spacer(1, 20))

    # Footer
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#6b7280'),
        alignment=1  # Center alignment
    )
    story.append(Paragraph("Generated by Kina - Speech to Cognitive Insights", footer_style))

    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


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
        if diversity > 0.5:
            lex_pattern = "✅ Diverse vocabulary."
        else:
            lex_pattern = "⚠️ Repeated words or less variety."

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
        error_msg = f'Could not request results from Speech Recognition service: {e}'
        return {'success': False, 'error': error_msg}
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
            ("Japanese", "ja-JP"),
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
            st.text_area("", results['transcription'], height=120, disabled=True,
                         label_visibility="collapsed")

            # Create three columns for metrics
            metric_col1, metric_col2, metric_col3 = st.columns(3)

            with metric_col1:
                st.markdown("#### 😊 Sentiment")
                if results['sentiment']['polarity'] > 0.1:
                    polarity_emoji = "😊"
                elif results['sentiment']['polarity'] > -0.1:
                    polarity_emoji = "😐"
                else:
                    polarity_emoji = "😔"

                st.metric(f"{polarity_emoji} Polarity",
                          f"{results['sentiment']['polarity']:.2f}",
                          help="−1 = negative, +1 = positive")
                st.metric("Subjectivity",
                          f"{results['sentiment']['subjectivity']:.2f}",
                          help="0 = objective, 1 = subjective")

            with metric_col2:
                st.markdown("#### 📚 Lexical Diversity")
                st.metric("Total Words", results['lexical']['total_words'])
                st.metric("Unique Words", results['lexical']['unique_words'])
                st.metric("Diversity Score",
                          f"{results['lexical']['diversity_score']:.2f}",
                          help="Higher is better (0-1 scale)")
                if results['lexical']['diversity_score'] > 0.5:
                    st.success(results['lexical']['pattern'])
                else:
                    st.warning(results['lexical']['pattern'])

            with metric_col3:
                st.markdown("#### 🧠 Complexity")
                avg_len = results['complexity']['avg_sentence_length']
                st.metric("Avg Sentence Length", f"{avg_len:.1f} words")
                st.metric("Conjunctions", results['complexity']['conjunction_count'])
                st.metric("Recording Duration", f"{results['duration']:.1f}s")
                if "✅" in results['complexity']['feedback']:
                    st.success(results['complexity']['feedback'])
                else:
                    st.warning(results['complexity']['feedback'])

            # Calculate and display cognitive score
            cognitive_results = calculate_cognitive_score(results)

            st.markdown("---")
            st.markdown("### 🎯 Cognitive Health Assessment")

            # Main score display
            score_col1, score_col2, score_col3 = st.columns([2, 1, 1])

            with score_col1:
                overall_score = cognitive_results['overall_score']
                st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 20px;
                        border-radius: 15px;
                        text-align: center;
                        margin: 10px 0;
                    ">
                        <h2 style="margin: 0; font-size: 48px;">{overall_score}</h2>
                        <p style="margin: 5px 0; font-size: 18px;">Overall Cognitive Score</p>
                        <p style="margin: 0; font-size: 14px; opacity: 0.9;">Out of 100</p>
                    </div>
                """, unsafe_allow_html=True)

            with score_col2:
                cognitive_age = cognitive_results['cognitive_age']
                st.markdown(f"""
                    <div style="
                        background: #f8f9fa;
                        padding: 20px;
                        border-radius: 15px;
                        text-align: center;
                        margin: 10px 0;
                        border: 2px solid #e9ecef;
                    ">
                        <h3 style="margin: 0; color: #495057; font-size: 32px;">{cognitive_age}</h3>
                        <p style="margin: 5px 0; color: #6c757d; font-size: 16px;">Estimated</p>
                        <p style="margin: 0; color: #6c757d; font-size: 16px;">Cognitive Age</p>
                    </div>
                """, unsafe_allow_html=True)

            with score_col3:
                risk_color = cognitive_results['risk_color']
                risk_level = cognitive_results['risk_level']
                st.markdown(f"""
                    <div style="
                        background: #f8f9fa;
                        padding: 20px;
                        border-radius: 15px;
                        text-align: center;
                        margin: 10px 0;
                        border: 2px solid #e9ecef;
                    ">
                        <h3 style="margin: 0; color: #495057; font-size: 24px;">{risk_color}</h3>
                        <p style="margin: 5px 0; color: #6c757d; font-size: 14px;">{risk_level}</p>
                        <p style="margin: 0; color: #6c757d; font-size: 12px;">Risk Level</p>
                    </div>
                """, unsafe_allow_html=True)

            # Risk description
            if cognitive_results['overall_score'] >= 80:
                st.success(f"🎉 {cognitive_results['risk_description']}")
            elif cognitive_results['overall_score'] >= 65:
                st.info(f"ℹ️ {cognitive_results['risk_description']}")
            elif cognitive_results['overall_score'] >= 50:
                st.warning(f"⚠️ {cognitive_results['risk_description']}")
            else:
                st.error(f"🚨 {cognitive_results['risk_description']}")

            # Component breakdown
            with st.expander("📊 Detailed Score Breakdown"):
                breakdown_col1, breakdown_col2 = st.columns(2)

                with breakdown_col1:
                    lexical_score = cognitive_results['component_scores']['lexical_diversity']
                    fluency_score = cognitive_results['component_scores']['speech_fluency']
                    words_per_sec = cognitive_results['words_per_second']

                    st.metric("Lexical Diversity", f"{lexical_score:.1f}/100")
                    st.metric("Speech Fluency", f"{fluency_score:.1f}/100")
                    st.caption(f"Speaking rate: {words_per_sec} words/second")

                with breakdown_col2:
                    complexity_score = cognitive_results['component_scores']['sentence_complexity']
                    emotion_score = cognitive_results['component_scores']['emotional_expression']

                    st.metric("Sentence Complexity", f"{complexity_score:.1f}/100")
                    st.metric("Emotional Expression", f"{emotion_score:.1f}/100")

            # Recommendations
            if cognitive_results['recommendations']:
                with st.expander("💡 Personalized Recommendations"):
                    for rec in cognitive_results['recommendations']:
                        st.write(f"• {rec}")

            # Download button for analysis results
            st.markdown("---")

            col_download, col_info = st.columns([1, 2])

            with col_download:
                try:
                    # Generate PDF report
                    pdf_data = generate_pdf_report(results, language[0])

                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_data,
                        file_name=f"kina_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        help="Download a detailed PDF report of your speech analysis"
                    )
                except Exception as e:
                    st.error(f"Error generating PDF: {e}")
                    # Fallback to text report
                    report_content = f"""KINA Speech Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Recording Duration: {results['duration']:.1f} seconds

=== TRANSCRIPTION ===
{results['transcription']}

=== SENTIMENT ANALYSIS ===
Polarity Score: {results['sentiment']['polarity']:.2f} (−1 = negative, +1 = positive)
Subjectivity Score: {results['sentiment']['subjectivity']:.2f} (0 = objective, 1 = subjective)

=== LEXICAL DIVERSITY ===
Total Words: {results['lexical']['total_words']}
Unique Words: {results['lexical']['unique_words']}
Diversity Score: {results['lexical']['diversity_score']:.2f} (Unique / Total)
Pattern Assessment: {results['lexical']['pattern']}

=== SENTENCE COMPLEXITY ===
Average Sentence Length: {results['complexity']['avg_sentence_length']:.1f} words
Conjunction Count: {results['complexity']['conjunction_count']}
Complexity Feedback: {results['complexity']['feedback']}

=== TECHNICAL INFO ===
Language: {language[0]}
Audio File: {results['audio_path']}
"""
                    st.download_button(
                        label="📥 Download Text Report",
                        data=report_content,
                        file_name=f"kina_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        help="Download a detailed text report of your speech analysis"
                    )

            with col_info:
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
