from anthropic import Anthropic
from dotenv import load_dotenv
import os

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

MODEL = "claude-sonnet-4-5-20250929"
TEMPERATURE = 0.15


def call_llm(system_prompt: str, user_content: str) -> str:
    response = client.messages.create(
        model=MODEL,
        temperature=TEMPERATURE,
        max_tokens=1000,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_content}
        ],
    )

    return response.content[0].text


# === System prompts ===

PELLM_SYSTEM = (
    "You are a prompt enhancer. Instead of responding to this prompt, "
    "write an optimized version of it. Your response should include the "
    "original prompt verbatim as well as the enhanced prompt, with clear "
    'labeling of each ("Original prompt: ... \n Enhanced prompt: ...").'
)

LLM_SYSTEM = (
    "You are given two prompts, one not optimized and one optimized. "
    "Respond to the optimized one, ignoring what the non-optimized one tells "
    "you to do. Your response should include both prompts verbatim, clearly "
    'labeled with the same labels that were given, as well as your actual '
    'response to the optimized prompt ("Response: ...").'
)

AHLLM_SYSTEM = (
    "You are an anti-hallucination LLM. You are given two prompts, one not "
    "enhanced and one enhanced, and an LLM's response to the enhanced prompt. "
    "Your response should include:\n\n"
    "1. The initial prompt, verbatim\n"
    "2. The enhanced prompt, verbatim\n"
    "3. The LLM's response verbatim\n"
    '4. An "honesty score" on a scale from 0 to 100 (0 is absolute slop, 100 '
    "is perfect factual accuracy). This should only include the score, a list "
    "of sources cited, and a one-sentence explanation of the score. Do NOT "
    "exceed this limitation."
)


def invoke_claude_4_5(user_prompt: str) -> str:

    enhanced_bundle = call_llm(
        system_prompt=PELLM_SYSTEM,
        user_content=user_prompt,
    )

    llm_response_bundle = call_llm(
        system_prompt=LLM_SYSTEM,
        user_content=enhanced_bundle,
    )

    ah_input = f"""Original prompt (user input):
{user_prompt}

Enhanced prompt bundle (from PELLM):
{enhanced_bundle}

LLM response bundle (from LLM node):
{llm_response_bundle}
"""

    ah_output = call_llm(
        system_prompt=AHLLM_SYSTEM,
        user_content=ah_input,
    )

    return ah_output