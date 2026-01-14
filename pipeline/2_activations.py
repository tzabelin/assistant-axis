#!/usr/bin/env python3
"""
Extract activations from response JSONL files.

This script loads responses from per-role JSONL files and extracts mean response
activations for each conversation, saving them as .pt files per role.

Usage:
    uv run scripts/2_activations.py \
        --model google/gemma-2-27b-it \
        --responses_dir outputs/gemma-2-27b/responses \
        --output_dir outputs/gemma-2-27b/activations
"""

import argparse
import gc
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import jsonlines
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from assistant_axis.internals import ProbingModel, ConversationEncoder, ActivationExtractor, SpanMapper

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_responses(responses_file: Path) -> List[dict]:
    """Load responses from JSONL file."""
    responses = []
    with jsonlines.open(responses_file, 'r') as reader:
        for entry in reader:
            responses.append(entry)
    return responses


def extract_activations_batch(
    pm: ProbingModel,
    conversations: List[List[Dict[str, str]]],
    layers: List[int],
    batch_size: int = 16,
    max_length: int = 2048,
) -> List[Optional[torch.Tensor]]:
    """Extract mean response activations for a batch of conversations."""
    encoder = ConversationEncoder(pm.tokenizer, pm.model_name)
    extractor = ActivationExtractor(pm, encoder)
    span_mapper = SpanMapper(pm.tokenizer)

    all_activations = []
    num_conversations = len(conversations)

    for batch_start in range(0, num_conversations, batch_size):
        batch_end = min(batch_start + batch_size, num_conversations)
        batch_conversations = conversations[batch_start:batch_end]

        # Use ActivationExtractor.batch_conversations to get activations
        batch_activations, batch_metadata = extractor.batch_conversations(
            batch_conversations,
            layer=layers,
            max_length=max_length,
        )

        # batch_activations shape: (num_layers, batch_size, max_seq_len, hidden_size)

        # Build spans for this batch
        _, batch_spans, span_metadata = encoder.build_batch_turn_spans(batch_conversations)

        # Use SpanMapper to get per-turn mean activations
        # Returns list of tensors, each (num_turns, num_layers, hidden_size)
        conv_activations_list = span_mapper.map_spans(batch_activations, batch_spans, batch_metadata)

        # For each conversation, we want the assistant turn activations
        # In single-turn conversations: turn 0 = user, turn 1 = assistant
        for conv_acts in conv_activations_list:
            if conv_acts.numel() == 0:
                all_activations.append(None)
                continue

            # conv_acts shape: (num_turns, num_layers, hidden_size)
            # For single-turn: (2, num_layers, hidden_size) - take turn 1 (assistant)
            # For multi-turn: take odd indices (assistant turns)
            if conv_acts.shape[0] >= 2:
                # Take the last assistant turn (index 1 for single-turn)
                assistant_act = conv_acts[1::2]  # All assistant turns
                if assistant_act.shape[0] > 0:
                    # Take mean across all assistant turns, transpose to (num_layers, hidden_size)
                    mean_act = assistant_act.mean(dim=0).cpu()  # (num_layers, hidden_size)
                    all_activations.append(mean_act)
                else:
                    all_activations.append(None)
            else:
                all_activations.append(None)

        # Cleanup
        del batch_activations
        if (batch_start // batch_size) % 5 == 0:
            torch.cuda.empty_cache()

    return all_activations


def main():
    parser = argparse.ArgumentParser(description="Extract activations from responses")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name")
    parser.add_argument("--responses_dir", type=str, required=True, help="Directory with response JSONL files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for .pt files")
    parser.add_argument("--layers", type=str, default="all", help="Layers to extract (all or comma-separated)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--skip_existing", action="store_true", help="Skip roles with existing output")
    parser.add_argument("--roles", nargs="+", help="Specific roles to process")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    responses_dir = Path(args.responses_dir)

    # Load model using ProbingModel
    logger.info(f"Loading model: {args.model}")
    pm = ProbingModel(args.model)

    # Determine layers
    n_layers = len(pm.get_layers())
    logger.info(f"Model has {n_layers} layers")

    if args.layers == "all":
        layers = list(range(n_layers))
    else:
        layers = [int(x.strip()) for x in args.layers.split(",")]

    logger.info(f"Extracting {len(layers)} layers")

    # Get response files
    response_files = sorted(responses_dir.glob("*.jsonl"))
    logger.info(f"Found {len(response_files)} response files")

    # Filter roles if specified
    if args.roles:
        response_files = [f for f in response_files if f.stem in args.roles]

    for response_file in tqdm(response_files, desc="Processing roles"):
        role = response_file.stem
        output_file = output_dir / f"{role}.pt"

        # Skip if exists
        if args.skip_existing and output_file.exists():
            continue

        # Load responses
        responses = load_responses(response_file)
        if not responses:
            continue

        # Extract conversations and metadata
        conversations = []
        metadata = []
        for resp in responses:
            conversations.append(resp["conversation"])
            metadata.append({
                "prompt_index": resp["prompt_index"],
                "question_index": resp["question_index"],
                "label": resp["label"],
            })

        logger.info(f"Processing {role}: {len(conversations)} conversations")

        # Extract activations
        activations_list = extract_activations_batch(
            pm=pm,
            conversations=conversations,
            layers=layers,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )

        # Build activation dict
        activations_dict = {}
        for i, (act, meta) in enumerate(zip(activations_list, metadata)):
            if act is not None:
                key = f"q{meta['question_index']}_p{meta['prompt_index']}"
                activations_dict[key] = act

        # Save
        if activations_dict:
            torch.save(activations_dict, output_file)
            logger.info(f"Saved {len(activations_dict)} activations for {role}")

        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()

    logger.info("Done!")


if __name__ == "__main__":
    main()
