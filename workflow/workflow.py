from openai import OpenAI

client = OpenAI()

MODEL = "gpt-5-chat-latest"  # same as in your Dify DSL
TEMPERATURE = 0.7


def call_llm(system_prompt: str, user_content: str) -> str:
    """
    Generic helper to call the chat model with a system prompt + user message.
    """
    response = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    return response.choices[0].message.content


# === System prompts from your DSL ===

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


def run_workflow(user_prompt: str) -> str:
    """
    This mirrors your Dify workflow:

    User Input -> PELLM -> LLM -> AHLLM -> final answer
    """

    # --- Step 1: PELLM (Prompt Enhancer) ---
    enhanced_bundle = call_llm(
        system_prompt=PELLM_SYSTEM,
        user_content=user_prompt,
    )
    # `enhanced_bundle` should contain:
    #   Original prompt: ...
    #   Enhanced prompt: ...

    # --- Step 2: LLM (Respond to optimized prompt) ---
    # In Dify, this node would see the two prompts produced above.
    # Easiest way to mirror that is to pass the whole text through.
    llm_response_bundle = call_llm(
        system_prompt=LLM_SYSTEM,
        user_content=enhanced_bundle,
    )
    # `llm_response_bundle` should contain both prompts verbatim + "Response: ..."

    # --- Step 3: AHLLM (Anti-hallucination checker) ---
    # This node expects: two prompts (non-enhanced + enhanced) + LLM response.
    # We'll assemble that explicitly to make sure it's clear.
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

    # This corresponds to your Answer node's `{{#17651248826940.text#}}`
    return ah_output


if __name__ == "__main__":
    # Simple CLI usage example
    while True:
        try:
            user_prompt = input("Enter your prompt (or 'quit'): ")
        except EOFError:
            break

        if user_prompt.strip().lower() in {"quit", "exit"}:
            break

        result = run_workflow(user_prompt)
        print("\n=== Final Output (AHLLM) ===\n")
        print(result)
        print("\n" + "=" * 40 + "\n")