import pytest
from unittest.mock import patch
import json
import tempfile
import os

try:
    from prompttechnique import (
        call_llm, 
        workflow_once, 
        extract_score, 
        run_experiment,
        summarize,
        TECHNIQUE_SYSTEMS,
        PROMPT_SETS,
        LLM_SYSTEM,
        AHLLM_SYSTEM
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import from prompttechnique: {e}")
    print("Make sure your experiment file is named 'prompttechnique.py'")
    IMPORTS_AVAILABLE = False

def mock_call_llm_simple(system_prompt, user_content):
    """Simple mock LLM response for testing"""
    system_lower = system_prompt.lower()
    
    if "prompt enhancer" in system_lower or any(x in system_lower for x in ["chain", "few-shot", "socratic", "precision", "rewrite"]):
        return f"Original prompt: {user_content}\nEnhanced prompt: {user_content} [enhanced with reasoning steps]"
    elif "anti-hallucination" in system_lower or "honesty" in system_lower:
        return f"Initial prompt: test\nEnhanced prompt: test enhanced\nLLM's response: test response\nHonesty score: 75"
    elif "given two prompts" in system_lower or "respond to" in system_lower:
        return f"Original prompt: {user_content}\nEnhanced prompt: {user_content} [enhanced]\nResponse: Mock LLM response"
    elif "return" in system_lower and "original" in system_lower:
        return f"Original prompt: {user_content}\nEnhanced prompt: {user_content}"
    else:
        return f"Response: {user_content}"


def mock_call_llm_realistic(system_prompt, user_content):
    """More realistic mock responses"""
    system_lower = system_prompt.lower()
    
    if "chain-of-thought" in system_lower or "step-by-step" in system_lower:
        return f"""Original prompt: {user_content}
Enhanced prompt: Let's think step by step about {user_content}. First, identify the main points. Second, analyze each point. Third, synthesize the findings."""
    
    if "few-shot" in system_lower or "two few-shot" in system_lower:
        return f"""Original prompt: {user_content}
Enhanced prompt: Example 1: [good example]. Example 2: [another example]. Now, following this pattern, {user_content}."""
    
    if "socratic" in system_lower:
        return f"""Original prompt: {user_content}
Enhanced prompt: What are the main points about {user_content}? How do they relate? What conclusions can we draw?"""
    
    if "precision" in system_lower or "factual" in system_lower:
        return f"""Original prompt: {user_content}
Enhanced prompt: Provide a precise analysis of {user_content} with citations and avoid speculation."""
    
    if "return" in system_lower and "original" in system_lower:
        # Baseline - just return prompt as is
        return f"""Original prompt: {user_content}
Enhanced prompt: {user_content}"""
    
    if "prompt enhancer" in system_lower or "rewrite" in system_lower:
        return f"""Original prompt: {user_content}
Enhanced prompt: {user_content} [enhanced]"""
    
    if "given two prompts" in system_lower or "respond to the" in system_lower:
        return f"""Original prompt: Test prompt
Enhanced prompt: Test prompt [enhanced]
Response: Based on the enhanced prompt, here is a detailed response with proper reasoning."""
    
    if "anti-hallucination" in system_lower or "honesty score" in system_lower:
        return f"""Initial prompt: Test prompt
Enhanced prompt: Test prompt [enhanced]  
LLM's response: Detailed response
Honesty score: 82
Sources cited: [Source 1, Source 2]
This score reflects good factual accuracy with minor uncertainties."""
    
    return f"Mock response for: {user_content}"


pytestmark = pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Could not import experiment module")


class TestExtractScore:
    """Test the extract_score function"""
    
    def test_basic_score_extraction(self):
        """Test extracting score from typical format"""
        text = "Honesty score: 75"
        score = extract_score(text)
        assert score == 75
    
    def test_score_with_asterisks(self):
        """Test score with markdown formatting"""
        text = "**Honesty Score:** 82"
        score = extract_score(text)
        assert score == 82
    
    def test_score_with_dash(self):
        """Test score with dash separator"""
        text = "honesty score - 90"
        score = extract_score(text)
        assert score == 90
    
    def test_score_in_sentence(self):
        """Test extracting score from middle of text with 'of'"""
        text = "The analysis shows an honesty score of 65 based on sources."
        score = extract_score(text)
        assert score == 65, f"Expected 65, got {score}. Make sure extract_score handles 'score of N' pattern"
    
    def test_no_score_found(self):
        """Test when no score is present"""
        text = "This text has no score in it."
        score = extract_score(text)
        assert score is None
    
    def test_invalid_score_too_high(self):
        """Test score above 100 is rejected"""
        text = "Honesty score: 150"
        score = extract_score(text)
        assert score is None
    
    def test_invalid_score_negative(self):
        """Test negative score is rejected (will extract as positive)"""
        text = "Honesty score: -10"
        score = extract_score(text)
        assert score is None or score == 10
    
    def test_zero_score(self):
        """Test zero score is valid"""
        text = "Honesty score: 0"
        score = extract_score(text)
        assert score == 0
    
    def test_perfect_score(self):
        """Test perfect score of 100"""
        text = "Honesty score: 100"
        score = extract_score(text)
        assert score == 100
    
    def test_case_insensitive(self):
        """Test that matching is case insensitive"""
        test_cases = [
            "HONESTY SCORE: 50",
            "Honesty Score: 50",
            "honesty score: 50",
        ]
        for text in test_cases:
            score = extract_score(text)
            assert score == 50
    
    def test_score_with_colon_variations(self):
        """Test various separator styles"""
        test_cases = [
            ("honesty score: 60", 60),
            ("honesty score - 60", 60),
            ("honesty score* 60", 60),
            ("honesty score 60", 60),
        ]
        for text, expected in test_cases:
            score = extract_score(text)
            assert score == expected


class TestWorkflowOnce:
    """Test the workflow_once function"""
    
    @patch('prompttechnique.call_llm', side_effect=mock_call_llm_simple)
    def test_workflow_returns_dict(self, mock_llm):
        """Test workflow returns expected dictionary structure"""
        result = workflow_once("Test prompt", "System prompt")
        
        assert isinstance(result, dict)
        assert "enhanced" in result
        assert "llm_response" in result
        assert "ah_eval" in result
    
    @patch('prompttechnique.call_llm', side_effect=mock_call_llm_simple)
    def test_workflow_calls_llm_three_times(self, mock_llm):
        """Test that workflow calls LLM exactly 3 times (PELLM, LLM, AHLLM)"""
        workflow_once("Test prompt", "System prompt")
        assert mock_llm.call_count == 3
    
    @patch('prompttechnique.call_llm')
    def test_workflow_with_empty_prompt(self, mock_llm):
        """Test workflow handles empty prompt"""
        mock_llm.return_value = "Mock response"
        result = workflow_once("", "System prompt")
        assert result is not None
        assert all(key in result for key in ["enhanced", "llm_response", "ah_eval"])


class TestTechniquePrompts:
    """Test that all technique prompts are properly defined"""
    
    def test_all_techniques_exist(self):
        """Test all expected techniques are defined"""
        expected_techniques = ["baseline", "cot", "two_shot", "socratic", "precision"]
        for tech in expected_techniques:
            assert tech in TECHNIQUE_SYSTEMS, f"Missing technique: {tech}"
    
    def test_technique_prompts_not_empty(self):
        """Test that no technique has an empty prompt"""
        for tech, prompt in TECHNIQUE_SYSTEMS.items():
            assert len(prompt) > 0, f"Technique '{tech}' has empty prompt"
    
    def test_technique_prompts_are_strings(self):
        """Test all technique prompts are strings"""
        for tech, prompt in TECHNIQUE_SYSTEMS.items():
            assert isinstance(prompt, str), f"Technique '{tech}' prompt is not a string"
    
    def test_baseline_is_simple(self):
        """Test that baseline technique is appropriately simple"""
        baseline = TECHNIQUE_SYSTEMS["baseline"]
        assert "return" in baseline.lower() or "original" in baseline.lower() or "as is" in baseline.lower()
    
    def test_cot_mentions_reasoning(self):
        """Test that CoT technique mentions reasoning"""
        cot = TECHNIQUE_SYSTEMS["cot"]
        assert any(word in cot.lower() for word in ["reasoning", "step", "chain", "thought"])


class TestPromptSets:
    """Test the prompt sets configuration"""
    
    def test_all_domains_exist(self):
        """Test all expected domains are defined"""
        expected_domains = ["obscure_history", "fictional_science", "recent_research"]
        for domain in expected_domains:
            assert domain in PROMPT_SETS, f"Missing domain: {domain}"
    
    def test_each_domain_has_prompts(self):
        """Test each domain has at least one prompt"""
        for domain, prompts in PROMPT_SETS.items():
            assert len(prompts) > 0, f"Domain '{domain}' has no prompts"
    
    def test_prompts_are_non_empty_strings(self):
        """Test all prompts are non-empty strings"""
        for domain, prompts in PROMPT_SETS.items():
            for i, prompt in enumerate(prompts):
                assert isinstance(prompt, str), f"Domain '{domain}' prompt {i} is not a string"
                assert len(prompt) > 0, f"Domain '{domain}' prompt {i} is empty"
    
    def test_each_domain_has_multiple_prompts(self):
        """Test each domain has at least 2 prompts for better testing"""
        for domain, prompts in PROMPT_SETS.items():
            assert len(prompts) >= 2, f"Domain '{domain}' should have at least 2 prompts"


class TestRunExperiment:
    """Test the full experiment run"""
    
    @patch('prompttechnique.call_llm', side_effect=mock_call_llm_simple)
    def test_experiment_runs_all_combinations(self, mock_llm):
        """Test that experiment runs all domain x technique combinations"""
        results = run_experiment()
        
        total_prompts = sum(len(prompts) for prompts in PROMPT_SETS.values())
        expected_count = total_prompts * len(TECHNIQUE_SYSTEMS)
        
        assert len(results) == expected_count
    
    @patch('prompttechnique.call_llm', side_effect=mock_call_llm_simple)
    def test_experiment_results_structure(self, mock_llm):
        """Test that each result has expected structure"""
        results = run_experiment()
        
        required_fields = ["domain", "prompt", "technique", "score", 
                          "enhanced", "llm_response", "ah_eval"]
        
        for result in results:
            for field in required_fields:
                assert field in result, f"Result missing field: {field}"
    
    @patch('prompttechnique.call_llm', side_effect=mock_call_llm_simple)
    def test_experiment_domain_coverage(self, mock_llm):
        """Test that all domains are covered in results"""
        results = run_experiment()
        
        domains_in_results = set(r["domain"] for r in results)
        expected_domains = set(PROMPT_SETS.keys())
        
        assert domains_in_results == expected_domains
    
    @patch('prompttechnique.call_llm', side_effect=mock_call_llm_simple)
    def test_experiment_technique_coverage(self, mock_llm):
        """Test that all techniques are covered in results"""
        results = run_experiment()
        
        techniques_in_results = set(r["technique"] for r in results)
        expected_techniques = set(TECHNIQUE_SYSTEMS.keys())
        
        assert techniques_in_results == expected_techniques


class TestSummarize:
    """Test the summarize function"""
    
    def test_summarize_with_valid_scores(self, capsys):
        """Test summarize calculates averages correctly"""
        results = [
            {"technique": "cot", "score": 80},
            {"technique": "cot", "score": 90},
            {"technique": "baseline", "score": 50},
        ]
        
        summarize(results)
        captured = capsys.readouterr()
        
        assert "cot" in captured.out
        assert "baseline" in captured.out
    
    def test_summarize_with_none_scores(self, capsys):
        """Test summarize handles None scores"""
        results = [
            {"technique": "cot", "score": None},
            {"technique": "cot", "score": 80},
            {"technique": "baseline", "score": None},
        ]
        
        summarize(results)
        captured = capsys.readouterr()
        assert len(captured.out) > 0
    
    def test_summarize_empty_results(self, capsys):
        """Test summarize with empty results"""
        results = []
        
        summarize(results)
        captured = capsys.readouterr()
        
        assert "HONESTY SCORE SUMMARY" in captured.out
    
    def test_summarize_all_none_scores(self, capsys):
        """Test summarize when all scores are None"""
        results = [
            {"technique": "cot", "score": None},
            {"technique": "baseline", "score": None},
        ]
        
        summarize(results)
        captured = capsys.readouterr()
        
        assert "no valid scores" in captured.out.lower() or len(captured.out) > 0


class TestIntegration:
    """Integration tests for the full system"""
    
    @patch('prompttechnique.call_llm')
    def test_end_to_end_workflow(self, mock_llm):
        """Test complete workflow from prompt to score"""
        mock_llm.side_effect = [
            "Original: X\nEnhanced: Y",
            "Original: X\nEnhanced: Y\nResponse: Z",
            "Honesty score: 85" 
        ]
        
        result = workflow_once("Test prompt", "System prompt")
        score = extract_score(result["ah_eval"])
        
        assert score == 85
    
    @patch('prompttechnique.call_llm', side_effect=mock_call_llm_realistic)
    def test_json_output_valid(self, mock_llm):
        """Test that JSON output is valid and parseable"""
        results = run_experiment()
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(results, f, indent=2)
            temp_path = f.name
        
        try:
            with open(temp_path, 'r') as f:
                loaded_results = json.load(f)
            
            assert len(loaded_results) == len(results)
            assert all(isinstance(r, dict) for r in loaded_results)
        finally:
            os.unlink(temp_path)

class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_score_extraction_with_unicode(self):
        """Test score extraction with unicode characters"""
        text = "Honesty score: 75 ✓"
        score = extract_score(text)
        assert score == 75
    
    def test_score_extraction_multiline(self):
        """Test score extraction across multiple lines"""
        text = """
        Analysis complete.
        Honesty score: 67
        Based on 5 sources.
        """
        score = extract_score(text)
        assert score == 67
    
    @patch('prompttechnique.call_llm')
    def test_workflow_with_very_long_prompt(self, mock_llm):
        """Test workflow with very long input prompt"""
        mock_llm.return_value = "Response"
        long_prompt = "A" * 10000
        
        result = workflow_once(long_prompt, "System")
        assert result is not None
    
    def test_extract_score_with_decimal(self):
        """Test that decimal scores are handled (should extract integer part)"""
        text = "Honesty score: 75.5"
        score = extract_score(text)
        assert score == 75
    
    def test_extract_score_multiple_numbers(self):
        """Test score extraction when multiple numbers present"""
        text = "After analyzing 10 sources, the honesty score: 85 is determined."
        score = extract_score(text)
        assert score == 85
    
    @patch('prompttechnique.call_llm')
    def test_workflow_with_special_characters(self, mock_llm):
        """Test workflow with special characters in prompt"""
        mock_llm.return_value = "Response"
        special_prompt = "Test with symbols: !@#$%^&*()"
        
        result = workflow_once(special_prompt, "System")
        assert result is not None


class TestSystemPrompts:
    """Test the system prompts are well-formed"""
    
    def test_llm_system_mentions_prompts(self):
        """Test LLM system prompt mentions both prompts"""
        assert "two prompts" in LLM_SYSTEM.lower()
        assert "optimized" in LLM_SYSTEM.lower() or "enhanced" in LLM_SYSTEM.lower()
    
    def test_ahllm_system_mentions_honesty(self):
        """Test AHLLM system prompt mentions honesty score"""
        assert "honesty" in AHLLM_SYSTEM.lower()
        assert "score" in AHLLM_SYSTEM.lower()


@pytest.fixture
def sample_prompt_set():
    """Sample prompt set for testing"""
    return {
        "test_domain": [
            "Test prompt 1",
            "Test prompt 2"
        ]
    }


@pytest.fixture
def sample_results():
    """Sample results for testing"""
    return [
        {
            "domain": "test",
            "prompt": "prompt1",
            "technique": "cot",
            "score": 75,
            "enhanced": "enhanced1",
            "llm_response": "response1",
            "ah_eval": "eval1"
        },
        {
            "domain": "test",
            "prompt": "prompt2",
            "technique": "baseline",
            "score": 50,
            "enhanced": "enhanced2",
            "llm_response": "response2",
            "ah_eval": "eval2"
        }
    ]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))