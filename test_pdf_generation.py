#!/usr/bin/env python3
"""
Simple test script to verify PDF generation works correctly.
This script generates a sample PDF report and saves it to verify functionality.
"""

import os
from datetime import datetime
from streamlit_app import generate_pdf_report

def create_sample_results():
    """Create sample analysis results for testing."""
    return {
        'success': True,
        'transcription': '''This is a sample transcription for testing the PDF generation functionality.
It contains multiple sentences with various punctuation marks!
How does the system handle different types of content?
Let's find out by testing with this comprehensive example.''',
        'raw_text': 'This is a sample transcription for testing the PDF generation functionality. It contains multiple sentences with various punctuation marks! How does the system handle different types of content? Let\'s find out by testing with this comprehensive example.',
        'duration': 25.7,
        'sentiment': {
            'polarity': 0.15,
            'subjectivity': 0.45
        },
        'lexical': {
            'total_words': 32,
            'unique_words': 28,
            'diversity_score': 0.875,
            'pattern': '✅ Diverse vocabulary.'
        },
        'complexity': {
            'avg_sentence_length': 8.0,
            'conjunction_count': 2,
            'feedback': '⚠️ Simple or flat sentence structure.'
        },
        'audio_path': 'recordings/sample_test_recording.wav'
    }

def test_pdf_generation():
    """Test PDF generation and save to file."""
    print("🧪 Testing PDF Generation...")

    # Create sample data
    results = create_sample_results()
    language = "English (US)"

    try:
        # Generate PDF
        print("📄 Generating PDF report...")
        pdf_data = generate_pdf_report(results, language)

        # Verify PDF data
        if not isinstance(pdf_data, bytes):
            raise ValueError("PDF data should be bytes")

        if not pdf_data.startswith(b'%PDF'):
            raise ValueError("Generated data is not a valid PDF")

        if len(pdf_data) < 1000:
            raise ValueError("PDF seems too small, might be corrupted")

        # Save to file for manual inspection
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"test_kina_report_{timestamp}.pdf"

        with open(filename, 'wb') as f:
            f.write(pdf_data)

        print(f"✅ PDF generated successfully!")
        print(f"📁 Saved as: {filename}")
        print(f"📊 File size: {len(pdf_data):,} bytes")

        # Verify file was created
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            print(f"✅ File verification: {file_size:,} bytes on disk")
        else:
            print("❌ File was not created on disk")
            return False

        return True

    except Exception as e:
        print(f"❌ PDF generation failed: {e}")
        return False

def test_edge_cases():
    """Test PDF generation with edge cases."""
    print("\n🧪 Testing Edge Cases...")

    edge_cases = [
        {
            'name': 'Empty transcription',
            'results': {
                'success': True,
                'transcription': '',
                'raw_text': '',
                'duration': 0.0,
                'sentiment': {'polarity': 0.0, 'subjectivity': 0.0},
                'lexical': {'total_words': 0, 'unique_words': 0, 'diversity_score': 0.0, 'pattern': '⚠️ No words detected.'},
                'complexity': {'avg_sentence_length': 0.0, 'conjunction_count': 0, 'feedback': '⚠️ No sentences detected.'},
                'audio_path': 'recordings/empty.wav'
            }
        },
        {
            'name': 'Special characters',
            'results': {
                'success': True,
                'transcription': 'Hello! 🎤 This contains émojis and spéciál characters: àáâãäå',
                'raw_text': 'Hello! 🎤 This contains émojis and spéciál characters: àáâãäå',
                'duration': 5.2,
                'sentiment': {'polarity': 0.3, 'subjectivity': 0.2},
                'lexical': {'total_words': 8, 'unique_words': 8, 'diversity_score': 1.0, 'pattern': '✅ Diverse vocabulary.'},
                'complexity': {'avg_sentence_length': 8.0, 'conjunction_count': 1, 'feedback': '⚠️ Simple or flat sentence structure.'},
                'audio_path': 'recordings/special_chars.wav'
            }
        }
    ]

    success_count = 0
    for case in edge_cases:
        try:
            print(f"  Testing: {case['name']}")
            pdf_data = generate_pdf_report(case['results'], "English (US)")

            if isinstance(pdf_data, bytes) and pdf_data.startswith(b'%PDF') and len(pdf_data) > 500:
                print(f"  ✅ {case['name']}: Success")
                success_count += 1
            else:
                print(f"  ❌ {case['name']}: Invalid PDF data")
        except Exception as e:
            print(f"  ❌ {case['name']}: {e}")

    print(f"\n📊 Edge case results: {success_count}/{len(edge_cases)} passed")
    return success_count == len(edge_cases)

def main():
    """Run all PDF generation tests."""
    print("🎤 KINA PDF Generation Test Suite")
    print("=" * 40)

    # Test basic PDF generation
    basic_test_passed = test_pdf_generation()

    # Test edge cases
    edge_cases_passed = test_edge_cases()

    # Summary
    print("\n" + "=" * 40)
    print("📋 Test Summary:")
    print(f"  Basic PDF Generation: {'✅ PASS' if basic_test_passed else '❌ FAIL'}")
    print(f"  Edge Cases: {'✅ PASS' if edge_cases_passed else '❌ FAIL'}")

    if basic_test_passed and edge_cases_passed:
        print("\n🎉 All tests passed! PDF generation is working correctly.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit(main())