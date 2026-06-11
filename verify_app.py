#!/usr/bin/env python3
"""
Simple verification script to check if the Streamlit app can be imported and run.
"""

def test_imports():
    """Test that all required imports work correctly."""
    print("🔍 Testing imports...")
    
    try:
        # Test reportlab imports
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from io import BytesIO
        print("✅ ReportLab imports successful")
        
        # Test streamlit app imports
        from streamlit_app import generate_pdf_report, format_text, lexical_diversity, sentence_complexity
        print("✅ Streamlit app function imports successful")
        
        # Test other required imports
        import streamlit as st
        import speech_recognition as sr
        from pydub import AudioSegment
        from textblob import TextBlob
        print("✅ All other imports successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_pdf_generation():
    """Test basic PDF generation functionality."""
    print("\n📄 Testing PDF generation...")
    
    try:
        from streamlit_app import generate_pdf_report
        
        # Sample data
        sample_results = {
            'success': True,
            'transcription': 'This is a test transcription.',
            'raw_text': 'This is a test transcription.',
            'duration': 5.0,
            'sentiment': {'polarity': 0.1, 'subjectivity': 0.3},
            'lexical': {'total_words': 5, 'unique_words': 5, 'diversity_score': 1.0, 'pattern': '✅ Diverse vocabulary.'},
            'complexity': {'avg_sentence_length': 5.0, 'conjunction_count': 0, 'feedback': '⚠️ Simple structure.'},
            'audio_path': 'test.wav'
        }
        
        # Generate PDF
        pdf_data = generate_pdf_report(sample_results, "English (US)")
        
        # Verify PDF
        if isinstance(pdf_data, bytes) and pdf_data.startswith(b'%PDF') and len(pdf_data) > 1000:
            print("✅ PDF generation successful")
            print(f"📊 PDF size: {len(pdf_data):,} bytes")
            return True
        else:
            print("❌ PDF generation failed - invalid data")
            return False
            
    except Exception as e:
        print(f"❌ PDF generation error: {e}")
        return False

def test_core_functions():
    """Test core analysis functions."""
    print("\n🧪 Testing core functions...")
    
    try:
        from streamlit_app import format_text, lexical_diversity, sentence_complexity
        
        test_text = "Hello world! This is a test sentence."
        
        # Test format_text
        formatted = format_text(test_text)
        assert isinstance(formatted, str)
        print("✅ format_text works")
        
        # Test lexical_diversity
        score, total, unique = lexical_diversity(test_text)
        assert isinstance(score, float) and isinstance(total, int) and isinstance(unique, int)
        print("✅ lexical_diversity works")
        
        # Test sentence_complexity
        avg_len, conj_count, feedback = sentence_complexity(test_text)
        assert isinstance(avg_len, float) and isinstance(conj_count, int) and isinstance(feedback, str)
        print("✅ sentence_complexity works")
        
        return True
        
    except Exception as e:
        print(f"❌ Core function error: {e}")
        return False

def main():
    """Run all verification tests."""
    print("🎤 KINA App Verification")
    print("=" * 30)
    
    # Run tests
    imports_ok = test_imports()
    pdf_ok = test_pdf_generation()
    functions_ok = test_core_functions()
    
    # Summary
    print("\n" + "=" * 30)
    print("📋 Verification Summary:")
    print(f"  Imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"  PDF Generation: {'✅ PASS' if pdf_ok else '❌ FAIL'}")
    print(f"  Core Functions: {'✅ PASS' if functions_ok else '❌ FAIL'}")
    
    if imports_ok and pdf_ok and functions_ok:
        print("\n🎉 All verifications passed! The app is ready to run.")
        print("\n🚀 To start the Streamlit app, run:")
        print("   streamlit run streamlit_app.py")
        return 0
    else:
        print("\n⚠️  Some verifications failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())