_This project has been created as part of the 42 curriculum by mlagutin._

# Call Me Maybe

## Short Description

Call Me Maybe is a 42 project that translates natural-language prompts into
structured function calls.

The program receives:

- a JSON file containing available function definitions
- a JSON file containing user prompts

For each prompt, it selects the most likely function and decodes the parameters
needed for that function. The final result is written as JSON, not as prose.

The project uses the provided `llm_sdk` with a small language model. The model is
used for scoring choices, while the program keeps the final output constrained
and validated.

## What The Program Does

For each input prompt, the pipeline:

1. Loads and validates function definitions.
2. Loads and validates prompts.
3. Uses the LLM to select one function from the available definitions.
4. Extracts candidate argument values from the prompt.
5. Scores candidate values with the LLM.
6. Builds a structured function-call result.
7. Validates the result with Pydantic.
8. Writes all successful results to a JSON output file.

Example output item:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {
    "a": 2.0,
    "b": 3.0
  }
}
```

## Requirements

- Python 3.10
- `uv`
- the provided `llm_sdk`
- project dependencies from `pyproject.toml`

Main direct dependencies include:

- `pydantic`
- `rich`
- `numpy`
- `pytest`, `flake8`, and `mypy` for development checks

The project interacts with the language model only through the provided
`llm_sdk`

The default model is:

```txt
Qwen/Qwen3-0.6B
```

## Installation

Install dependencies with:

```sh
make install
```

or directly with:

```sh
uv sync
```

The `Makefile` places the `uv` and Hugging Face caches under `~/goinfre`, which
is useful in the 42 environment.

## Usage

Run the program with the default input and output paths:

```sh
uv run python -m src
```

The same command is available through:

```sh
make run
```

The default paths are:

```txt
data/input/functions_definition.json
data/input/function_calling_tests.json
data/output/function_calling_results.json
```

Custom paths can be provided:

```sh
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

A custom model name can also be passed to `llm_sdk.Small_LLM_Model`:

```sh
uv run python -m src --model_name Qwen/Qwen3-0.6B
```

## Input Files

### Function Definitions

The function definition file must contain a JSON array. Each item is validated as
a `FunctionDefinition` Pydantic model.

Supported parameter types are:

- `string`
- `number`
- `integer`
- `boolean`
- `object`

Example:

```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": {
        "type": "number"
      },
      "b": {
        "type": "number"
      }
    },
    "returns": {
      "type": "number"
    }
  }
]
```

### Prompt Input

The prompt file must contain a JSON array. Each item can be either:

- a string prompt
- an object with a string `prompt` field

Example:

```json
[
  "What is the sum of 2 and 3?",
  {
    "prompt": "Greet Alice"
  }
]
```

## Output Format

By default, results are written to:

```txt
data/output/function_calling_results.json
```

The output file contains a JSON array of structured results:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {
      "a": 2.0,
      "b": 3.0
    }
  }
]
```

The output JSON contains only structured results. Terminal tables and progress
messages are not written into the JSON file.

If a prompt cannot be routed or decoded, it is reported in the terminal and is
not added to the output results.

## Architecture Overview

Important source files:

```txt
src/__main__.py          CLI entry point
src/config.py            default paths and model name
src/models.py            Pydantic input and output models
src/io_utils.py          JSON reading, validation, and writing
src/pipeline.py          main orchestration
src/selector.py          function selection with LLM logits
src/decoder.py           schema-guided argument decoding
src/extractor.py         generic prompt value extraction helpers
src/candidate_scorer.py  LLM scoring for argument candidates
src/render.py            Rich terminal output
src/tokenizer_utils.py   tokenizer compatibility helpers
src/errors.py            project-specific exceptions
```

The main pipeline is implemented in `src/pipeline.py`. It loads inputs, creates
the model, selects a function, decodes arguments, validates results, and writes
the output file.

## Function Selection

Function selection is implemented in `src/selector.py`.

The selector builds a deterministic routing prompt from the user request and the
available function definitions. Each candidate function name is tokenized and
scored with model logits.

To avoid giving duplicate credit to shared prefixes, the selector detects a
common token prefix between candidate names and scores only the discriminative
part of each candidate name. The highest-scoring function is selected.

If the score gap between the best and second-best function is too small, the
selector raises a `SelectionError`. This keeps uncertain routing out of the final
JSON output.

## Argument Decoding

Argument decoding is implemented in `src/decoder.py`.

The decoder does not ask the model to freely generate JSON. Instead, it uses the
selected function schema to decide which kind of values are needed, extracts
candidate values from the prompt, and asks the model to score those candidates.

Candidate extraction is implemented in `src/extractor.py` and includes generic
helpers for:

- numbers
- quoted strings
- booleans
- file paths
- simple regex hints
- replacement hints
- words near known parameter keywords
- text after a colon

Candidate scoring is implemented in `src/candidate_scorer.py`. For each
parameter, the scorer builds a small context and compares the log-probability of
candidate values. The best-scoring candidate is used as the argument value.

Numeric values are converted to `float` for `number` parameters and to `int` for
`integer` parameters. Boolean candidates are decoded from simple words such as
`true`, `false`, `yes`, `no`, `enable`, and `disable`.

Object parameters are currently accepted by the schema, but full nested object
decoding is not implemented. The current decoder returns an empty object for
parameters of type `object`.

## Error Handling

The project defines custom exceptions in `src/errors.py` for common failure
cases:

- file read errors
- file write errors
- invalid JSON
- schema validation errors
- function selection errors
- argument decoding errors

Input files are validated before processing. Output items are represented with
the `FunctionCallResult` Pydantic model before being written.

If the whole input configuration is invalid, the program prints an error to
stderr and exits with status `1`.

If a specific prompt cannot be selected or decoded, the program reports it in
the terminal and continues with the next prompt.

## Visualization

The project uses `rich` terminal tables to show how decisions are made while the
program runs.

The terminal output includes:

- prompt progress
- function routing scores
- argument candidate scores
- the selected function and decoded parameters
- the final output path and number of written results

These visualizations are for debugging and evaluation only. They are not part of
the JSON output file.

## Testing And Linting

Tests are implemented with `pytest` in the `tests/` directory.

The test suite covers the main modules, including:

- Pydantic models
- JSON loading and writing helpers
- function selection
- argument decoding
- candidate scoring
- extraction helpers
- rendering helpers
- end-to-end pipeline behavior

The following checks pass for this project:

```sh
make test
make lint
make lint-strict
```

## Makefile Commands

Available commands:

```sh
make install       # prepare caches and install dependencies with uv
make run           # run uv run python -m src
make debug         # run the program under pdb
make test          # run pytest
make lint          # run flake8 and mypy
make lint-strict   # run flake8 and mypy --strict
make clean         # remove local Python/tool caches
make space         # show disk usage information
```

## Implemented Bonus-Like Features

The project does not claim that every bonus is implemented. The following
bonus-like features are present:

- custom model name support through `--model_name`
- comprehensive pytest coverage for the current implementation
- Rich visualization of function routing and argument scoring
- candidate-based argument scoring instead of free-form JSON generation
- graceful handling of invalid files, validation errors, uncertain selection,
  and failed argument decoding

## Limitations / Not Implemented

Current limitations:

- full nested object decoding is not implemented
- a full custom tokenizer implementation is not implemented
- batching and caching performance optimizations are not implemented
- argument extraction is based on generic candidate helpers, so it is limited by
  the values that can be found in the prompt
- unsupported or ambiguous prompts may be skipped instead of forced into the
  output

## Project Status

The project is functional for the required 42 function-calling workflow:

- default run command: `uv run python -m src`
- default model: `Qwen/Qwen3-0.6B`
- default output: `data/output/function_calling_results.json`
- output format: structured JSON results with `prompt`, `name`, and
  `parameters`
- validation: Pydantic models for inputs and outputs
- tests and linting: `make test`, `make lint`, and `make lint-strict` pass

