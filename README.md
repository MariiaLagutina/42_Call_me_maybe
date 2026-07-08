# Call Me Maybe

_This project has been created as part of the 42 curriculum by mlagutin._

## Short Description

Call Me Maybe translates natural-language prompts into structured function-call
records.

The program receives a list of available function definitions and a list of user
prompts. For each prompt, it selects a function, decodes its parameters, validates
the result, and writes JSON output.

The project uses the provided `llm_sdk` and model logits. It does not ask the LLM
to freely write arbitrary JSON. Instead, it constrains the decision space to
known function names, extracted candidate values, or token masks for simple value
fallbacks.

## Requirements

- Python 3.10
- `uv`
- the provided `llm_sdk`
- dependencies from `pyproject.toml`

Main direct project dependencies:

- `pydantic`
- `rich`
- `numpy`
- `pytest`, `flake8`, and `mypy` for checks

The language model dependencies are installed so the provided `llm_sdk` can run,
but the project code interacts with the model only through `llm_sdk`.

Default model:

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

The `Makefile` stores `uv` and Hugging Face caches under `~/goinfre`, which is
useful in the 42 environment.

## Usage

Run with default paths:

```sh
uv run python -m src
```

or:

```sh
make run
```

Default files:

```txt
data/input/functions_definition.json
data/input/function_calling_tests.json
```

Custom paths:

```sh
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

Custom model name:

```sh
uv run python -m src --model_name Qwen/Qwen3-0.6B
```

## Input Files

The function definition file must contain a JSON array. Each function is
validated with a Pydantic `FunctionDefinition` model.

Supported parameter types:

- `string`
- `number`
- `integer`
- `boolean`
- `object`

Example function definition:

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

The prompt file must contain a JSON array. Each item can be a string or an
object with a string `prompt` field.

Example prompt file:

```json
[
  "What is the sum of 2 and 3?",
  {
    "prompt": "Greet Alice"
  }
]
```

## Output Format

By default, output is generated at runtime at:

```txt
data/output/function_calling_results.json
```

`data/output` is ignored by git and should not be committed.

The output file is a JSON array of structured results:

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

The output JSON contains only structured results. It does not include prose,
terminal logs, Rich tables, or model explanations.

If a prompt cannot be selected or decoded, it is reported in the terminal and is
not forced into the output file.

## Architecture Overview

Main files:

```txt
src/__main__.py                 CLI entry point
src/config.py                   default paths and model name
src/models.py                   Pydantic models
src/io_utils.py                 JSON loading, validation, and writing
src/pipeline.py                 main orchestration
src/selector.py                 function selection with logits
src/decoder.py                  argument decoding flow
src/extractor.py                prompt-value extraction helpers
src/candidate_scorer.py         candidate value scoring with logits
src/constrained_choice.py       constrained finite-candidate scoring helper
src/constrained_value_decoder.py token-mask fallback value decoder
src/tokenizer_vocab.py          tokenizer vocabulary loading
src/token_constraints.py        token sets for constrained values
src/logit_utils.py              small logit helper functions
src/tokenizer_utils.py          tokenizer compatibility helpers
src/render.py                   Rich terminal visualization
src/errors.py                   project exceptions
```

The pipeline loads and validates input files, creates the model with
`llm_sdk.Small_LLM_Model`, selects a function, decodes arguments, validates the
final `FunctionCallResult`, and writes JSON.

Pydantic validation is the final safety layer before output is written.

## Function Selection

Function selection is implemented in `src/selector.py`.

The selector builds a routing prompt from the user request and available
function definitions. It then scores candidate function names with model logits.

Candidate names are encoded with the model tokenizer. If names share a token
prefix, the shared prefix is skipped for scoring so the comparison focuses on
the tokens that distinguish the candidates.

If the best function is not sufficiently ahead of the second-best function, a
`SelectionError` is raised and that prompt is skipped.

## Argument Decoding

Argument decoding is implemented in `src/decoder.py`.

The decoder does not ask the LLM to freely generate JSON. It uses a hybrid
constrained approach:

1. The selected function schema decides which parameters are needed.
2. Generic helpers in `src/extractor.py` extract explicit values from the prompt.
3. Extracted candidates are scored with logits in `src/candidate_scorer.py`.
4. If no explicit candidate exists and tokenizer constraints are available, the
   token-mask fallback in `src/constrained_value_decoder.py` is used.

Candidate extraction currently handles common explicit values such as numbers,
quoted strings, booleans, file paths, simple regex hints, replacement hints, and
some parameter-name-specific text patterns.

For `object` parameters, the current decoder returns an empty object. Full nested
object decoding is not implemented.

## Token Vocabulary And Token Masks

The token-mask fallback is used for simple values when no extracted candidate is
available and tokenizer vocabulary files can be loaded from the model.

Related modules:

- `src/tokenizer_vocab.py`
- `src/token_constraints.py`
- `src/logit_utils.py`
- `src/constrained_choice.py`
- `src/constrained_value_decoder.py`

The fallback constrains the possible next tokens:

- strings use safe string-body tokens and can stop on a quote token
- numbers use number-compatible tokens only
- booleans are constrained by scoring `true` and `false`

This is not full unrestricted token-by-token JSON generation. The fallback only
generates simple parameter values under masks built from the tokenizer
vocabulary.

If tokenizer vocabulary information is not available, the fallback is disabled
and missing candidates become decoding errors.

## Error Handling

Project exceptions are defined in `src/errors.py`.

Handled cases include:

- missing or unreadable files
- empty files
- invalid JSON
- invalid input schemas
- uncertain function selection
- failed argument decoding
- output write errors

Invalid input configuration exits with an error message and status `1`.
Per-prompt selection or decoding failures are reported in the terminal, and the
program continues with the remaining prompts.

## Visualization

Rich terminal output is used to make peer evaluation easier.

The terminal can show:

- prompt progress
- function routing scores
- argument candidate scores
- selected function calls
- skipped prompts and reasons
- final output path and result count

These messages are terminal-only. They are not written into the output JSON.

## Testing And Linting

Tests are implemented with `pytest` in `tests/`.

The tests cover model validation, JSON IO, function selection, extraction,
candidate scoring, constrained choice helpers, token constraints, constrained
value fallback behavior, rendering helpers, and pipeline behavior.

The following commands pass:

```sh
make test
make lint
make lint-strict
```

## Makefile Commands

```sh
make install       # prepare caches and install dependencies
make run           # run uv run python -m src
make debug         # run the program under pdb
make test          # run pytest
make lint          # run flake8 and mypy
make lint-strict   # run flake8 and mypy --strict
make clean         # remove Python/tool caches
make space         # show disk usage information
```

## Implemented Extras

The project does not claim that every optional feature is implemented. Current
extras include:

- custom model name support through `--model_name`
- pytest coverage
- Rich terminal visualization
- logits-based function selection
- schema-guided candidate scoring
- vocabulary/token-mask fallback decoding for simple values
- graceful error handling

## Limitations

- This is not a full general-purpose function-calling framework.
- Full nested object decoding is not implemented; object parameters currently
  decode to `{}`.
- Batching and caching optimizations are not implemented.
- Candidate extraction depends on generic prompt-value helpers.
- Ambiguous or unsupported prompts may be skipped or reported instead of forced
  into the output.
- Token-mask fallback decoding is limited to simple strings, numbers, integers,
  and booleans.

## Challenges Faced

One challenge was balancing practical reliability with the constrained decoding
requirements of the subject. The project uses schema-guided candidate scoring
for explicit values and a token-masked fallback decoder for simple values such
as strings, numbers, integers, and booleans.

Another challenge was keeping a clean separation between my project code and the
provided `llm_sdk`. The model dependencies are required for the SDK to run, but
the source code interacts with the model only through the public SDK interface.

## Resources

Resources used:
- provided llm_sdk
- Hugging Face model page for Qwen/Qwen3-0.6B, used to understand the model referenced by the subject and llm_sdk
- Python documentation for json, argparse, pathlib, and typing
- Pydantic documentation
- pytest, flake8, and mypy documentation
- Rich documentation for terminal tables


## AI usage

AI tools were used as learning support during development: to discuss design
alternatives, improve wording, review type-checking issues, and better
understand constrained decoding with small language models.

The Pydantic models, function selection flow, argument decoding logic,
token-mask fallback, terminal visualization, tests, and project-specific extras
were reviewed, adapted, and implemented by me.

I only kept code and design decisions that I understood and could explain.