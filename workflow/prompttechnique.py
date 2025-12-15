from openai import OpenAI
import re
import json

client = OpenAI(api_key="")

MODEL = "gpt-4-turbo"
# TEMPERATURE = 0.7


def call_llm(system_prompt: str, user_content: str) -> str:
    """
    Generic helper to call the chat model with a system prompt + user message.
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    return response.choices[0].message.content


PELLM_BASELINE_SYSTEM = (
    "Just return the original prompt as is"
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


TECHNIQUE_SYSTEMS = {
    "baseline": PELLM_BASELINE_SYSTEM,

    "cot": (
        "You are a prompt enhancer. Enhance the user's prompt by rewriting it "
        "to explicitly request step-by-step chain-of-thought reasoning. "
        "Output:\nOriginal prompt: ...\nEnhanced prompt: ..."
    ),

    "two_shot": (
        "You are a prompt enhancer. Add two few-shot examples demonstrating "
        "how the task should be done well, then rewrite the user's prompt in "
        "that style. Output original + enhanced prompt clearly labeled."
    ),

    "socratic": (
        "You are a prompt enhancer. Transform the prompt into a series of "
        "Socratic questions designed to guide correct reasoning. Output "
        "original + enhanced."
    ),

    "precision": (
        "You are a prompt enhancer. Rewrite the prompt to enforce factual "
        "precision, discourage hallucinations, and require citations. "
        "Output original + enhanced."
    ),
}


PROMPT_SETS = {
    "obscure_history": [
        "Summarize the role of General Hartmann in the Baltic Reorganization Act of 1764.",
        "Explain the consequences of the 1823 Vinberg Maritime Trade Reforms."
    ],

    "fictional_science": [
        "Describe the Moreschi Convergence Hypothesis in anthropology.",
        "Explain Perelman's 1991 Continuity Mapping Theorem and its applications."
    ],

    "recent_research": [
        "Summarize the findings of the January 2025 DARPA BioCompute paper on synthetic neurons.",
        "Explain the experimental results reported in the 2024 LHC forward-detector neutrino dataset."
    ]
}



def workflow_once(user_prompt: str, pel_system_prompt: str):
    """
    Runs the original workflow once using a chosen PELLM system prompt.
    LLM and AHLLM remain untouched.
    """

    # Step 1: PELLM
    enhanced_bundle = call_llm(
        system_prompt=pel_system_prompt,
        user_content=user_prompt,
    )

    # Step 2: LLM
    llm_response_bundle = call_llm(
        system_prompt=LLM_SYSTEM,
        user_content=enhanced_bundle,
    )

    # Step 3: AHLLM
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

    return {
        "enhanced": enhanced_bundle,
        "llm_response": llm_response_bundle,
        "ah_eval": ah_output,
    }

def extract_score(ah_text: str):
    pattern = r'honesty\s*score\s*[:\-\*]*\s*(?:of\s*)?(\d{1,3})'
    m = re.search(pattern, ah_text, re.IGNORECASE)
    if not m:
        return None
    score = int(m.group(1))
    if 0 <= score <= 100:
        return score
    return None
    
def run_experiment():
    results = []

    for domain, prompts in PROMPT_SETS.items():
        for prompt in prompts:
            for technique, technique_system in TECHNIQUE_SYSTEMS.items():

                print(f"\n--- Running domain={domain} | technique={technique} ---")

                out = workflow_once(prompt, technique_system)

                score = extract_score(out["ah_eval"])

                results.append({
                    "domain": domain,
                    "prompt": prompt,
                    "technique": technique,
                    "score": score,
                    "enhanced": out["enhanced"],
                    "llm_response": out["llm_response"],
                    "ah_eval": out["ah_eval"],
                })

    return results


def summarize(results):
    buckets = {}

    for r in results:
        tech = r["technique"]
        buckets.setdefault(tech, []).append(r["score"])

    print("\nHONESTY SCORE SUMMARY")
    for tech, scores in buckets.items():
        valid = [s for s in scores if s is not None]
        if valid:
            avg = sum(valid) / len(valid)
            print(f"{tech:15s} → {avg:5.1f}")
        else:
            print(f"{tech:15s} → no valid scores")


if __name__ == "__main__":
    print("Running PELLM prompting experiment…")
    results = run_experiment()
    summarize(results)

    # Save full output
    with open("experiment_results-test.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nFull results saved to experiment_results-test.json")