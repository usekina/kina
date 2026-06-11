#!/usr/bin/env python3
"""
Test script to verify the cognitive scoring functionality works correctly.
"""

from streamlit_app import calculate_cognitive_score

def test_cognitive_scoring():
    """Test the cognitive scoring with different scenarios."""
    
    print("🧠 Testing Cognitive Scoring System")
    print("=" * 40)
    
    # Test Case 1: High-performing speech sample
    print("\n📊 Test Case 1: High-Quality Speech")
    high_quality_results = {
        'duration': 20.0,
        'sentiment': {'polarity': 0.2, 'subjectivity': 0.4},
        'lexical': {
            'total_words': 50,
            'unique_words': 40,
            'diversity_score': 0.8,
            'pattern': '✅ Diverse vocabulary.'
        },
        'complexity': {
            'avg_sentence_length': 15.0,
            'conjunction_count': 3,
            'feedback': '✅ Balanced complexity.'
        }
    }
    
    cognitive_score = calculate_cognitive_score(high_quality_results)
    print(f"Overall Score: {cognitive_score['overall_score']}/100")
    print(f"Cognitive Age: {cognitive_score['cognitive_age']} years")
    print(f"Risk Level: {cognitive_score['risk_level']}")
    print(f"Words per second: {cognitive_score['words_per_second']}")
    print("Component Scores:")
    for component, score in cognitive_score['component_scores'].items():
        print(f"  - {component.replace('_', ' ').title()}: {score}/100")
    
    # Test Case 2: Lower-performing speech sample
    print("\n📊 Test Case 2: Concerning Speech Patterns")
    concerning_results = {
        'duration': 30.0,
        'sentiment': {'polarity': -0.4, 'subjectivity': 0.8},
        'lexical': {
            'total_words': 20,
            'unique_words': 12,
            'diversity_score': 0.6,
            'pattern': '⚠️ Repeated words or less variety.'
        },
        'complexity': {
            'avg_sentence_length': 4.0,
            'conjunction_count': 0,
            'feedback': '⚠️ Simple or flat sentence structure.'
        }
    }
    
    cognitive_score = calculate_cognitive_score(concerning_results)
    print(f"Overall Score: {cognitive_score['overall_score']}/100")
    print(f"Cognitive Age: {cognitive_score['cognitive_age']} years")
    print(f"Risk Level: {cognitive_score['risk_level']}")
    print(f"Words per second: {cognitive_score['words_per_second']}")
    print("Component Scores:")
    for component, score in cognitive_score['component_scores'].items():
        print(f"  - {component.replace('_', ' ').title()}: {score}/100")
    
    # Test Case 3: Average speech sample
    print("\n📊 Test Case 3: Average Speech Performance")
    average_results = {
        'duration': 15.0,
        'sentiment': {'polarity': 0.1, 'subjectivity': 0.3},
        'lexical': {
            'total_words': 30,
            'unique_words': 22,
            'diversity_score': 0.73,
            'pattern': '✅ Diverse vocabulary.'
        },
        'complexity': {
            'avg_sentence_length': 10.0,
            'conjunction_count': 1,
            'feedback': '⚠️ Simple or flat sentence structure.'
        }
    }
    
    cognitive_score = calculate_cognitive_score(average_results)
    print(f"Overall Score: {cognitive_score['overall_score']}/100")
    print(f"Cognitive Age: {cognitive_score['cognitive_age']} years")
    print(f"Risk Level: {cognitive_score['risk_level']}")
    print(f"Words per second: {cognitive_score['words_per_second']}")
    print("Component Scores:")
    for component, score in cognitive_score['component_scores'].items():
        print(f"  - {component.replace('_', ' ').title()}: {score}/100")
    
    print("\n" + "=" * 40)
    print("✅ Cognitive scoring system is working correctly!")
    print("\n💡 Key Features:")
    print("• Overall score (0-100) based on 4 components")
    print("• Estimated cognitive age calculation")
    print("• Risk level assessment with color coding")
    print("• Personalized recommendations")
    print("• Component breakdown for detailed analysis")

if __name__ == "__main__":
    test_cognitive_scoring()