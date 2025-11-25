"""
Generate bias prompts, generate responses, and extract activations.

This script combines prompt generation and activation extraction because we need:
1. Generate prompts
2. Run EXAONE to generate full responses
3. Extract activations at the ANSWER TOKEN position (not prompt last token)

Architecture aligned with korean-sparse-llm-features-open/script/gather_synthetic_activations.py

Supports two prompt formats:
- Generation format: Model generates demographic value directly (e.g., "남자", "여자")
- QA format: Model selects numbered option (e.g., "0", "1") that maps to demographic

Key difference from old approach:
- OLD: Extract from prompt's last token → Wrong!
- NEW: Extract from answer token position → Correct!
"""

import sys
import pickle
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.models import EXAONEWrapper
from src.utils import load_json, load_config
from src.utils.token_position import (
    estimate_token_location,
    extract_generated_demographic,
    extract_qa_answer,
    extract_qa_demographic
)
from src.utils.prompt_generation import format_options
from src.utils.demographic_utils import (
    validate_demographic_config,
    get_demographic_info,
    get_demographic_values,
    format_demographic_info
)


def generate_prompts(config, stage='pilot', qa_format=False):
    """
    Generate bias prompts from modifiers and templates.

    Args:
        config: Experiment configuration
        stage: 'pilot', 'medium', or 'full'
        qa_format: If True, use QA format templates with numbered options

    Returns:
        List of prompt dictionaries
    """
    # Load modifiers
    negative_mods = load_json(f'data/modifiers/{stage}_negative_ko.json')
    positive_mods = load_json(f'data/modifiers/{stage}_positive_ko.json')

    # Load templates (QA or generation format)
    templates_data = load_json('data/templates/korean_templates.json')
    template_key = f'{stage}_templates_qa' if qa_format else f'{stage}_templates'
    templates = templates_data[template_key]

    # Get demographic info
    demographic = config['data']['demographic']
    demographic_values = config['data']['demographic_values']

    # Pre-compute options string for QA format
    options_str = None
    if qa_format:
        # Strip leading spaces for display, keep original for mapping
        options_str = format_options(demographic_values)

    # Generate prompts
    prompts = []
    for modifier_type, modifiers in [('N', negative_mods), ('P', positive_mods)]:
        for modifier in modifiers:
            for template in templates:
                # For QA format, replace {Options} BEFORE calling .format()
                # because .format() requires ALL placeholders to be provided
                if qa_format and options_str is not None:
                    template_with_options = template.replace('{Options}', options_str)
                else:
                    template_with_options = template

                prompt = template_with_options.format(
                    Modifier=modifier,
                    Demographic_Dimension=demographic
                )

                prompts.append({
                    'prompt': prompt,
                    'modifier': modifier,
                    'modifier_type': modifier_type,
                    'demographic_dimension': demographic,
                    'demographic_values': demographic_values,
                    'num_options': len(demographic_values),
                    'qa_format': qa_format
                })

    return prompts


def main(args):
    # Load config
    config = load_config('configs/experiment_config.yaml')

    # Determine QA format from config or args
    qa_format = args.qa_format
    if 'qa_format' in config.get('data', {}):
        qa_format = config['data']['qa_format'].get('enabled', qa_format)

    # Override demographic if specified via command line
    if args.demographic:
        demographic = args.demographic
        demographic_values_config = get_demographic_values(demographic)
        config['data']['demographic'] = demographic
        config['data']['demographic_values'] = demographic_values_config
    else:
        demographic = config['data']['demographic']
        demographic_values_config = config['data']['demographic_values']

    print(f"=== Generating and Extracting Activations ===")
    print(f"Stage: {args.stage}")
    print(f"Demographic: {demographic}")
    print(f"Format: {'QA Multiple Choice (0-indexed)' if qa_format else 'Free-form Generation'}")

    is_valid, msg = validate_demographic_config(
        demographic,
        demographic_values_config,
        data_dir='data'
    )

    if not is_valid:
        print(f"\nERROR: Invalid demographic configuration!")
        print(f"   {msg}")
        print(f"\nPlease update configs/experiment_config.yaml")
        print(f"See data/demographic_dict_ko.json for valid options.")
        return

    print(f"Demographic configuration validated")
    print(f"\n{format_demographic_info(demographic, data_dir='data')}")

    # Get demographic values (strip leading spaces for comparison)
    demographic_values = [v.strip() for v in demographic_values_config]

    print(f"\nActive demographic values for this experiment:")
    for i, val in enumerate(demographic_values):
        print(f"  {i}. '{val}'" if qa_format else f"  {i+1}. '{val}'")

    # Load EXAONE
    print("\nLoading EXAONE model...")
    # Get primary device from devices list
    devices = config['model'].get('devices', ['cuda'])
    primary_device = devices[0] if isinstance(devices, list) else devices
    exaone = EXAONEWrapper(
        model_name=config['model']['name'],
        device=primary_device,
        dtype=config['model']['dtype']
    )
    print(f"Model loaded: {exaone.model_name}")
    print(f"Number of layers: {exaone.num_layers}")

    # Generate prompts
    print(f"\nGenerating {args.stage} prompts...")
    prompts = generate_prompts(config, stage=args.stage, qa_format=qa_format)
    print(f"Generated {len(prompts)} prompts")

    # Show example prompt
    if prompts:
        print(f"\nExample prompt:")
        print("-" * 40)
        print(prompts[0]['prompt'])
        print("-" * 40)

    # Prepare activation storage
    # This matches the format from korean-sparse-llm-features-open
    activations = {
        f'{args.stage}_input_ids': [],
        f'{args.stage}_labels': [],
        f'{args.stage}_prompts': [],
        f'{args.stage}_generated_texts': [],
        f'{args.stage}_residual_q1': [],
        f'{args.stage}_residual_q2': [],
        f'{args.stage}_residual_q3': [],
    }

    # Calculate layer indices (following korean-sparse-llm-features-open)
    num_layers = exaone.num_layers
    layer_q1 = int(num_layers * 0.25)
    layer_q2 = int(num_layers * 0.5)
    layer_q3 = int(num_layers * 0.75)

    print(f"\nLayer indices:")
    print(f"  Q1 (25%): Layer {layer_q1}")
    print(f"  Q2 (50%): Layer {layer_q2}")
    print(f"  Q3 (75%): Layer {layer_q3}")

    # Process each prompt
    print(f"\nGenerating responses and extracting activations...")
    print(f"This will take a while ({len(prompts)} prompts)...\n")

    errors = []
    for idx, prompt_data in enumerate(tqdm(prompts, desc="Processing prompts")):
        prompt = prompt_data['prompt']
        is_qa = prompt_data.get('qa_format', False)

        try:
            # 1. Generate full response from EXAONE
            generated_text = exaone.generate(
                prompt,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,  # Greedy decoding for consistency
                pad_token_id=exaone.tokenizer.eos_token_id
            )

            # 2. Extract answer based on format
            if is_qa:
                # QA format: Extract number answer and map to demographic
                answer_num, generated_demo, _ = extract_qa_demographic(
                    generated_text,
                    demographic_values_config  # Use original values with leading spaces
                )
                # For token position, look for the number token
                target_for_position = answer_num
            else:
                # Generation format: Extract demographic value directly
                generated_demo = extract_generated_demographic(
                    generated_text,
                    demographic_values
                )
                target_for_position = generated_demo

            # 3. Find token position of the answer
            # This is the KEY step that aligns with korean-sparse-llm-features-open
            tokens, answer_pos = estimate_token_location(
                text=generated_text,
                target=target_for_position,
                tokenizer=exaone.tokenizer,
                window_size=args.window_size,
                max_length=args.max_length
            )

            # 4. Extract hidden states at answer token position (NOT prompt end!)
            # Extract from multiple layers (following korean-sparse-llm-features-open)
            hidden_q1 = exaone.get_hidden_states_at_position(
                text=None,
                layer_idx=layer_q1,
                token_position=answer_pos,
                tokens=tokens
            )
            hidden_q2 = exaone.get_hidden_states_at_position(
                text=None,
                layer_idx=layer_q2,
                token_position=answer_pos,
                tokens=tokens
            )
            hidden_q3 = exaone.get_hidden_states_at_position(
                text=None,
                layer_idx=layer_q3,
                token_position=answer_pos,
                tokens=tokens
            )

            # 5. Store activations
            activations[f'{args.stage}_input_ids'].append(tokens[answer_pos])
            activations[f'{args.stage}_labels'].append(generated_demo)
            activations[f'{args.stage}_prompts'].append(prompt)
            activations[f'{args.stage}_generated_texts'].append(generated_text)
            activations[f'{args.stage}_residual_q1'].append(hidden_q1.detach().cpu())
            activations[f'{args.stage}_residual_q2'].append(hidden_q2.detach().cpu())
            activations[f'{args.stage}_residual_q3'].append(hidden_q3.detach().cpu())

        except Exception as e:
            error_msg = f"Prompt {idx}: {prompt}\nError: {str(e)}"
            errors.append(error_msg)
            print(f"\n[ERROR] {error_msg}")
            continue

    # Report errors if any
    if errors:
        print(f"\n\n{'='*60}")
        print(f"WARNING: {len(errors)} prompts failed during processing")
        print(f"{'='*60}")
        for error in errors[:5]:  # Show first 5 errors
            print(error)
            print("-" * 60)
        if len(errors) > 5:
            print(f"... and {len(errors) - 5} more errors")

    # Save activations
    print(f"\nSaving activations...")
    # Include demographic in output path for organization
    output_dir = Path(config['paths']['results_dir']) / args.stage / demographic
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stack tensors (following korean-sparse-llm-features-open format)
    num_successful = len(activations[f'{args.stage}_labels'])
    print(f"Successfully processed: {num_successful}/{len(prompts)} prompts")

    if num_successful == 0:
        print("ERROR: No prompts were successfully processed!")
        return

    stacked_activations = {
        f'{args.stage}_input_ids': torch.stack(activations[f'{args.stage}_input_ids']),
        f'{args.stage}_labels': activations[f'{args.stage}_labels'],
        f'{args.stage}_prompts': activations[f'{args.stage}_prompts'],
        f'{args.stage}_generated_texts': activations[f'{args.stage}_generated_texts'],
        f'{args.stage}_residual_q1': torch.cat(activations[f'{args.stage}_residual_q1'], dim=0),
        f'{args.stage}_residual_q2': torch.cat(activations[f'{args.stage}_residual_q2'], dim=0),
        f'{args.stage}_residual_q3': torch.cat(activations[f'{args.stage}_residual_q3'], dim=0),
    }

    # Verify shapes
    print(f"\nActivation shapes:")
    for key, val in stacked_activations.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: {val.shape}")
        else:
            print(f"  {key}: {len(val)} items")

    # Save to pickle file (same format as korean-sparse-llm-features-open)
    output_path = output_dir / 'activations.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(stacked_activations, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nSaved activations to: {output_path}")

    # Print label distribution
    unique_labels, counts = np.unique(
        activations[f'{args.stage}_labels'],
        return_counts=True
    )
    print(f"\nLabel distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"  {label}: {count} ({count/num_successful*100:.1f}%)")

    # Save summary statistics
    summary = {
        'stage': args.stage,
        'qa_format': qa_format,
        'demographic': config['data']['demographic'],
        'demographic_values': demographic_values,
        'total_prompts': len(prompts),
        'successful_prompts': num_successful,
        'failed_prompts': len(errors),
        'label_distribution': dict(zip(unique_labels.tolist(), counts.tolist())),
        'layer_indices': {
            'q1': layer_q1,
            'q2': layer_q2,
            'q3': layer_q3
        },
        'activation_shapes': {
            'input_ids': stacked_activations[f'{args.stage}_input_ids'].shape,
            'residual_q1': stacked_activations[f'{args.stage}_residual_q1'].shape,
            'residual_q2': stacked_activations[f'{args.stage}_residual_q2'].shape,
            'residual_q3': stacked_activations[f'{args.stage}_residual_q3'].shape,
        }
    }

    summary_path = output_dir / 'activation_summary.pkl'
    with open(summary_path, 'wb') as f:
        pickle.dump(summary, f)

    print(f"\nSaved summary to: {summary_path}")
    print(f"\n{'='*60}")
    print(f"✓ Activation extraction complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate bias prompts and extract activations at answer token positions"
    )
    parser.add_argument(
        '--stage',
        type=str,
        default='pilot',
        choices=['pilot', 'medium', 'full'],
        help='Experiment stage (determines data size)'
    )
    parser.add_argument(
        '--qa_format',
        action='store_true',
        default=False,
        help='Use QA multiple choice format with numbered options (0-indexed)'
    )
    parser.add_argument(
        '--window_size',
        type=int,
        default=10,
        help='Sliding window size for token position search'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=128,
        help='Maximum sequence length for tokenization'
    )
    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=5,
        help='Maximum new tokens to generate in response'
    )
    parser.add_argument(
        '--demographic',
        type=str,
        default=None,
        help='Demographic category (성별, 인종, 종교, etc.). Overrides config.'
    )
    args = parser.parse_args()

    main(args)
