# COMS 6998 Project: Complete & Trustworthy LLMs

This repository implements the **Complete & Trustworthy LLM** framework described in the accompanying project report. The core idea is a **three-agent LLM pipeline** designed to improve transparency and user trust without silently modifying model outputs:

1. **Prompt Enhancement LLM (PELLM)**  
   Rewrites and clarifies the user’s original prompt without answering it.

2. **Primary LLM**  
   Generates an answer strictly to the *enhanced* prompt and includes both the original and enhanced prompts verbatim in its response.

3. **Anti-Hallucination LLM (AHLLM)**  
   Evaluates the final answer and produces an **honesty score (0–100)** with a short justification and (when relevant) source citations.  
   Importantly, the AHLLM **does not edit or correct** the answer—it only assesses trustworthiness.

The pipeline is implemented consistently across multiple providers (OpenAI, Anthropic, Cohere) and is used both in an interactive web demo and in controlled prompt-strategy experiments.

---

## Repository Structure

```text
COMS_6998_Project/
├── README.md
├── requirements.txt
├── server.py
├── templates/
│   └── index.html
└── workflow/
    ├── gpt_5.py
    ├── anthropic.py
    ├── cohere.py
    ├── prompttechnique.py
    ├── prompt-technique-test.py
    ├── experiment_results-test.json
    └── plot.py
````

---

## Core Runtime Components

### `server.py` — Flask Backend API

`server.py` provides the backend for interacting with the three-agent pipeline.

* Serves the HTML UI at `GET /`.
* Exposes a JSON API at `POST /llm`.
* Accepts requests of the form:

  ```json
  {
    "model": "gpt_5 | anthropic | cohere",
    "prompt": "user prompt text"
  }
  ```
* Dispatches the request to the corresponding provider implementation in `workflow/`.
* Returns the **AHLLM evaluation output**, including the honesty score and explanation.
* Uses CORS to support local development and browser-based access.

This file is the main entry point for running the interactive system.

---

### `templates/index.html` — Web Interface

A minimal single-page interface for experimenting with the system.

* Allows the user to:

  * Select an LLM provider (GPT-5, Claude, or Cohere).
  * Enter a free-form prompt.
  * View the resulting output from the pipeline.
* Sends a `fetch()` request to `http://localhost:8112/llm`.
* Displays the returned text directly, making the honesty score and explanation visible to the user.

This UI is intentionally simple to highlight the model behavior rather than the interface.

---

### `requirements.txt` — Dependencies

Defines the Python dependencies required to run the project, including:

* Flask and Flask-CORS for the web server
* `python-dotenv` for API key management
* SDKs for OpenAI, Anthropic, and Cohere

---

## Workflow Modules: Three-Agent Pipeline

Each provider-specific module implements the same three-step architecture:

1. **Prompt Enhancement (PELLM)**
   Enhances clarity, constraints, and intent while preserving the original question.

2. **Answer Generation (LLM)**
   Answers only the enhanced prompt and includes:

   * Original prompt
   * Enhanced prompt
   * Generated answer

3. **Honesty Evaluation (AHLLM)**
   Produces:

   * An honesty score from 0–100
   * A short justification
   * Citations or uncertainty markers when applicable

This design ensures consistency across providers and makes experimental comparisons meaningful.

---

### `workflow/gpt_5.py` — OpenAI Implementation

* Implements the full pipeline using OpenAI’s API.
* Reads `OPENAI_API_KEY` from the environment.
* Defines explicit system prompts for:

  * PELLM
  * LLM
  * AHLLM
* Exposes a single entry point:

  ```python
  invoke_gpt_5(prompt: str) -> str
  ```
* Returns the AHLLM evaluation report as a string.

---

### `workflow/anthropic.py` — Anthropic (Claude) Implementation

* Mirrors the structure of `gpt_5.py`.
* Uses the Anthropic SDK and reads `ANTHROPIC_API_KEY`.
* Entry point:

  ```python
  invoke_claude_4_5(prompt: str) -> str
  ```
* Produces the same type of AHLLM evaluation output for consistency.

---

### `workflow/cohere.py` — Cohere Implementation

* Implements the same three-agent pipeline using Cohere’s chat API.
* Reads `COHERE_API_KEY` from the environment.
* Entry point:

  ```python
  invoke_cohere(prompt: str) -> str
  ```
* Ensures parity with OpenAI and Anthropic runs for fair comparison.

---

## Prompt-Strategy Experiments and Analysis

Beyond the interactive system, the repository contains code to evaluate **prompt enhancement strategies** under hallucination-prone conditions, as discussed in the report.

### `workflow/prompttechnique.py` — Experiment Runner

This file orchestrates controlled experiments across:

* **Prompt enhancement techniques**:

  * `baseline`
  * `cot` (chain-of-thought)
  * `two_shot`
  * `socratic`
  * `precision`
* **Prompt domains**:

  * Obscure historical facts
  * Fictional or speculative science
  * Recent or poorly documented research topics

Key functionality:

* `workflow_once(...)`
  Runs a single PELLM → LLM → AHLLM pass with a chosen technique.
* `extract_score(...)`
  Parses and validates the honesty score (0–100) from AHLLM output.
* `run_experiment()`
  Executes all combinations of domains, prompts, and techniques.
* `summarize()`
  Computes and prints average honesty scores per technique.

When run as a script, this module writes the full results to disk.

---

### `workflow/experiment_results-test.json` — Experiment Output

A serialized record of experiment runs, where each entry includes:

* Domain
* Prompt text
* Prompt enhancement technique
* Honesty score
* Full enhanced prompt
* LLM response
* AHLLM evaluation text

This file serves as the data source for analysis and plotting.

---

### `workflow/plot.py` — Visualization and Analysis

Loads `experiment_results-test.json` and produces:

* Honesty score distributions by technique
* Average improvement over the baseline strategy
* Domain × technique heatmaps for:

  * Raw honesty scores
  * Score improvements

These plots support the report’s conclusion that **precision prompting** yields the most consistent gains.

---

### `workflow/prompt-technique-test.py` — Tests

Provides unit and integration tests using `pytest`.

* Validates honesty score extraction and bounds.
* Ensures each pipeline run calls the LLM exactly three times.
* Confirms that all techniques and domains are exercised.
* Verifies that experiment outputs are JSON-serializable.

This file ensures the experimental framework is reproducible and robust.

---

## Step-by-Step Tutorial

This section provides a complete walkthrough for installing, configuring, running, and troubleshooting the project.

---

### 1. Installation Instructions

#### Prerequisites
- Python **3.9 or newer**
- `pip`
- At least one API key from:
  - OpenAI
  - Anthropic
  - Cohere

#### Clone the Repository
```bash
git clone https://github.com/hariharanep/COMS_6998_Project.git
cd COMS_6998_Project
````

#### Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Environment Setup Guide

This project uses environment variables to manage API keys securely.

#### Create a `.env` File

In the root directory, create a file named `.env`:

```text
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
COHERE_API_KEY=your_cohere_key_here
```

You only need to provide keys for the providers you intend to use.

#### Verify Environment Variables

On macOS / Linux:

```bash
echo $OPENAI_API_KEY
```

On Windows (PowerShell):

```powershell
echo $Env:OPENAI_API_KEY
```

---

### 3. Usage Examples and Demonstrations

#### A. Running the Web Demo

Start the Flask server:

```bash
python server.py
```

By default, the server runs at:

```
http://localhost:8112
```

Open this URL in a browser to access the interactive interface.

##### Example Workflow

1. Select a model provider (GPT-5, Claude, or Cohere)
2. Enter a prompt (e.g., a factual or ambiguous question)
3. Submit the prompt
4. Review the output, which includes:

   * The enhanced prompt
   * The model’s answer
   * The AHLLM honesty score and justification

---

#### B. Calling the API Directly

You can also interact with the backend via `curl` or any HTTP client.

```bash
curl -X POST http://localhost:8112/llm \
  -H "Content-Type: application/json" \
  -d '{
        "model": "gpt_5",
        "prompt": "Who first discovered the Tunguska event and what evidence did they use?"
      }'
```

The response will be the AHLLM evaluation report as plain text.

---

#### C. Running Prompt-Strategy Experiments

To reproduce the experimental results from the report:

```bash
python workflow/prompttechnique.py
```

This will:

* Run all prompt enhancement techniques across all prompt domains
* Print summary statistics to the console
* Write detailed results to:

  ```
  workflow/experiment_results-test.json
  ```

---

#### D. Generating Plots

After running the experiments:

```bash
python workflow/plot.py
```

This will generate visualizations comparing honesty scores and improvements across techniques and domains.

---

### 4. Troubleshooting Guide

#### Issue: `KeyError` or authentication failure from an API

**Cause:** Missing or invalid API key.
**Solution:**

* Ensure the correct key is present in `.env`
* Restart the server after updating environment variables

---

#### Issue: Server starts but UI cannot connect

**Cause:** CORS or incorrect URL.
**Solution:**

* Confirm the server is running on `http://localhost:8112`
* Ensure `index.html` is being served from Flask, not opened directly as a file

---

#### Issue: Honesty score not detected or shows as invalid

**Cause:** The AHLLM output format was not followed exactly.
**Solution:**

* This is expected occasionally with LLM variability
* The experiment code explicitly validates score bounds and will surface errors

---

#### Issue: Experiments run slowly

**Cause:** Large number of LLM calls across multiple techniques and domains.
**Solution:**

* Reduce the number of prompts or techniques in `prompttechnique.py`
* Run experiments selectively for a single provider

---

#### Issue: Tests fail

**Cause:** API calls not mocked or environment not isolated.
**Solution:**

* Ensure `pytest` is installed
* Tests are designed to mock LLM calls and should not require API keys