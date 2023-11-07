"""
Produces generations for the fill-in-the-middle benchmark from the SantaCoder
paper. Use analysis.py to compile the results into success rates.

The approach in this script is not the most robust approach possible: we
encode and decode the FIM format by looking for the textual representations of
the <|endoftext|> and FIM tokens. A more robust approach would look for the
token IDs directly in the output tensor. But, I think this approach is more
readable and the potential for error is low: the model would have to generate
 <|endoftext|> or the FIM tokens *as text* for this script to go wrong.
"""
import json
from pathlib import Path
import argparse
from more_itertools import chunked
from tqdm import tqdm
import torch
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

# These are the textual representations used in StarCoder models (*not*
# SantaCoder).
FIM_PREFIX = "<fim_prefix>"
FIM_MIDDLE = "<fim_middle>"
FIM_SUFFIX = "<fim_suffix>"
FIM_PAD = "<fim_pad>"
EOD = "<|endoftext|>"

GENERATION_ARGS = {
    "do_sample": True,
    "temperature": 0.2,
    "top_p": 0.95,
    "max_new_tokens": 25,
}

# We use this regex to remove the left-padding.
LEFT_PADDING_TEXT = re.compile("^(" + re.escape(EOD) + ")*")


def _fim_encode(mode: str, prefix: str, suffix: str) -> str:
    if mode == "PSM":
        return f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"
    elif mode == "PSMrepo":
        return f"{FIM_PREFIX}<reponame>jscarberry/pots<filename>test.py\n{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"
    elif mode == "SPMv2":
        # StarCoder and StarCoder 2 use SPMv2 and _not_ SPM.
        return f"{FIM_PREFIX}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}{prefix}"
    else:
        raise ValueError(f"Unknown mode {mode}")


def _fim_decode_psm(generation: str) -> str:
    """
    Given <fim_prefix>prefix<fim_suffix>suffix<fim_middle> returns middle.

    Does not address padding.
    """
    assert generation.startswith(
        FIM_PREFIX
    ), f"expected to start with {FIM_PREFIX}: {generation[:20]}"
    middle_index = generation.index(FIM_MIDDLE)
    return generation[middle_index + len(FIM_MIDDLE) :]


def _fim_decode_spmv2(generation: str, prefix: str) -> str:
    """
    Given <fim_prefix><fim_suffix>suffix<fim_middle>prefixmiddle, returns
    middle.

    Does not address padding.
    """
    assert generation.startswith(FIM_PREFIX + FIM_SUFFIX)
    middle_index = generation.index(FIM_MIDDLE)
    # This is just a sanity check.
    prefix_start = middle_index + len(FIM_MIDDLE)
    prefix_end = prefix_start + len(prefix)
    extracted_prefix = generation[prefix_start:prefix_end]
    assert extracted_prefix == prefix, f"SPMv2 error: {extracted_prefix} != {prefix}"
    return generation[middle_index + len(FIM_MIDDLE) + len(prefix) :]


def _fim_decode(mode: str, prefix: str, generation: str) -> str:
    """
    Decode a FIM result from the model and takes care of padding.
    """
    # Remove all the left-padding.
    generation = re.sub(LEFT_PADDING_TEXT, "", generation)
    if mode == "PSM" or mode == "PSMrepo":
        generation = _fim_decode_psm(generation)
    elif mode == "SPMv2":
        generation = _fim_decode_spmv2(generation, prefix)
    else:
        raise ValueError(f"Unknown mode {mode}")
    # Remove all the right-padding.
    try:
        right_pad_index = generation.index(EOD)
    except ValueError:
        right_pad_index = len(generation)
    return generation[:right_pad_index]


def _fill_in_the_middle(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prefix_suffix_tuples: List[Tuple[str, str]],
    mode: str,
) -> List[str]:
    prompts = [
        _fim_encode(mode, prefix, suffix) for prefix, suffix in prefix_suffix_tuples
    ]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
        return_token_type_ids=False,
    ).to("cuda")
    with torch.no_grad():
        output_tensors = model.generate(
            **inputs,
            **GENERATION_ARGS,
            pad_token_id=tokenizer.pad_token_id,
        )
    output_texts = tokenizer.batch_decode(
        output_tensors, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    return [
        _fim_decode(mode, prefix, text)
        for (text, (prefix, _)) in zip(output_texts, prefix_suffix_tuples)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", type=Path, required=True, help="Module name of the model to use"
    )
    parser.add_argument(
        "--batch-size", type=int, required=True, help="Batch size to use"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Output directory for results",
    )
    parser.add_argument(
        "--mode", choices=["SPMv2", "PSM", "PSMrepo"], required=True, help="Mode to use"
    )

    args = parser.parse_args()
    name = args.model_path.name

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, padding_side="left", trust_remote_code=True
    )
    tokenizer.pad_token = "<|endoftext|>"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.float16
    ).cuda()

    # Load existing results if any
    result_path = args.output_dir / f"fim-results-{args.mode}-{name}.jsonl"
    if result_path.exists():
        with result_path.open("rt") as f:
            results = [json.loads(line) for line in f]
    else:
        results = []

    # Load the problems and skip those that we already have results.
    with Path("benchmark.jsonl").open("rt") as f:
        problems = [json.loads(line) for line in f]
    problems = problems[len(results) :]

    problems = list(chunked(problems, args.batch_size))

    with result_path.open("at") as f:
        for batch in tqdm(
            problems, unit="Batch", desc="FIM inference", total=len(problems)
        ):
            pairs = [(p["prompt"], p["suffix"]) for p in batch]
            batch_results = _fill_in_the_middle(tokenizer, model, pairs, mode=args.mode)
            for problem, result in zip(batch, batch_results):
                problem["result"] = result
                problem["model"] = name
                problem["exact_match"] = (
                    result.strip() == problem["canonical_solution"].strip()
                )
                problem["fim_mode"] = args.mode
            for result in batch:
                f.write(json.dumps(result))
                f.write("\n")
                f.flush()


if __name__ == "__main__":
    main()
