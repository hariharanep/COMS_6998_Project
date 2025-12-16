from gpt_5 import invoke_gpt_5
import pytest

def test_invoke_gpt_5_happy_path():
    calls = []

    def fake_llm(system_prompt, user_content):
        calls.append((system_prompt, user_content))
        return f"OUTPUT({system_prompt[:10]})"

    result = invoke_gpt_5("hello world", llm_fn=fake_llm)

    assert len(calls) == 3
    assert calls[0][0].startswith("You are a prompt enhancer")
    assert calls[1][0].startswith("You are given two prompts")
    assert calls[2][0].startswith("You are an anti-hallucination")

    assert "OUTPUT(" in result

def test_pipeline_data_flow():
    outputs = {
        "PELLM": "ENHANCED PROMPT",
        "LLM": "LLM RESPONSE",
        "AH": "HONESTY SCORE: 95",
    }

    def fake_llm(system_prompt, user_content):
        if "You are a prompt enhancer." in system_prompt:
            return outputs["PELLM"]
        elif "You are given two prompts, one not optimized and one optimized." in system_prompt:
            return outputs["LLM"]
        elif "anti-hallucination" in system_prompt:
            assert outputs["PELLM"] in user_content
            assert outputs["LLM"] in user_content
            return outputs["AH"]

    result = invoke_gpt_5("test", llm_fn=fake_llm)
    assert result == outputs["AH"]

def test_llm_failure_propagates():
    def fake_llm(system_prompt, user_content):
        raise RuntimeError("LLM failed")

    result = invoke_gpt_5("test", llm_fn=fake_llm)
    assert result == "Error occurred. Please try again or change your prompt slightly before trying again."


