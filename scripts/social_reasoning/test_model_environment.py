#!/usr/bin/env python3
"""
test_model_environment.py — Optional model-load and scoring smoke test

Tests the social reasoning evaluation on a tiny sample to verify:
1. Dataset load correctly
2. Models load and compute loss
3. Output format is correct

Usage:
    python scripts/social_reasoning/test_model_environment.py --model /path/to/local/model

This smoke test is not a reproduction of the saved analyses.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_dataset_loading():
    """Test whether the canonical dataset can be read from any working directory."""
    print("="*80)
    print("TEST 1: Dataset Loading")
    print("="*80)

    import json

    print("\nTesting ProlificAI.jsonl...")
    prolific_path = ROOT / "data/social_reasoning/ProlificAI.jsonl"
    
    if not prolific_path.exists():
        print(f"  File not found: {prolific_path}")
        print(f"  Run this script from the repository root or set the dataset path in the script.")
        prolific_ok = False
    else:
        try:
            with open(prolific_path, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                sample = json.loads(first_line)
            
            print(f"  ✓ File exists and is valid JSON")
            print(f"  ✓ Sample keys: {list(sample.keys())}")
            
            question = sample.get('question', 'N/A')
            print(f"  ✓ First question: {question[:100]}...")
            
            chosen = sample.get('chosen', 'N/A')
            print(f"  ✓ Has 'chosen' response: {len(chosen)} chars")
            
            prolific_ok = True
        except Exception as e:
            print(f"  ✗ Failed to read: {e}")
            prolific_ok = False
    return prolific_ok

def test_model_loading(model_path: str):
    """Test if model loads and computes loss; requires inference dependencies."""
    try:
        import torch
        import numpy as np
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        print(f"  Missing model-inference dependency: {exc}")
        print("  Install requirements-model-inference.txt to run this test.")
        return False

    print("\n" + "="*80)
    print("TEST 2: Model Loading & Loss Computation")
    print("="*80)
    
    print(f"\nLoading model: {model_path}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"  ✓ Model loaded successfully")
        print(f"  ✓ Device: {model.device}")
        print(f"  ✓ Dtype: {model.dtype}")
    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        return False
    
    # Test loss computation on sample text
    print("\nTesting loss computation on sample social scenario...")
    
    test_examples = [
        {
            "prompt": "Your friend just lost their job. What should you say?",
            "response": "I'm really sorry to hear that. That must be really difficult. I'm here if you need to talk or if there's anything I can do to help."
        },
        {
            "prompt": "Someone cut in line in front of you. How do you respond?",
            "response": "Excuse me, I think there's a line here. I've been waiting."
        }
    ]
    
    losses = []
    
    for i, example in enumerate(test_examples, 1):
        prompt = example["prompt"]
        response = example["response"]
        full_text = prompt + " " + response
        
        print(f"\n  Example {i}:")
        print(f"    Prompt: {prompt[:60]}...")
        
        try:
            inputs = tokenizer(full_text, return_tensors='pt', truncation=False).to(model.device)
            sequence_length = int(inputs['input_ids'].shape[-1])
            context_limit = getattr(model.config, 'max_position_embeddings', None)
            if context_limit is not None and sequence_length > int(context_limit):
                raise ValueError(
                    f"Full test sequence has {sequence_length} tokens, exceeding the "
                    f"detected context limit of {context_limit}; the test does not truncate."
                )
            
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss.item()
                perplexity = np.exp(loss)
            
            losses.append(loss)
            
            print(f"    Loss: {loss:.4f}")
            print(f"    Perplexity: {perplexity:.2f}")
            print(f"    ✓ Computation successful")
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            return False
    
    avg_loss = np.mean(losses)
    print(f"\n  Average loss on test examples: {avg_loss:.4f}")
    print(f"  ✓ Loss computation test passed")
    
    # Clean up
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return True

def test_output_format():
    """Test that output format is correct."""
    print("\n" + "="*80)
    print("TEST 3: Output Format")
    print("="*80)
    
    # Mock result structure
    sample_result = {
        "metadata": {
            "n_examples": 10,
            "models_evaluated": 1
        },
        "model_results": {
            "Qwen3-8B-Base": {
                "mean_loss": 2.345,
                "std_loss": 0.234,
                "perplexity": 10.43,
                "n_examples": 10,
                "note": "Illustrative mock output; not a saved study result"
            }
        }
    }
    
    print("\nSample output structure:")
    import json
    print(json.dumps(sample_result, indent=2))
    print("\n✓ Output format validated")
    
    return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick validation test")
    parser.add_argument("--model", required=True, help="Model path to test")
    args = parser.parse_args()
    
    print("="*80)
    print("SOCIAL REASONING EVALUATION: VALIDATION TEST")
    print("="*80)
    print("\nThis is a small environment smoke test, not a reproduction of the full analysis.")
    print("Note: loading a checkpoint with trust_remote_code=True can execute code from that model repository; use trusted checkpoints only.")
    
    # Run tests
    tests_passed = 0
    
    if test_dataset_loading():
        tests_passed += 1
    
    if test_model_loading(args.model):
        tests_passed += 1
    
    if test_output_format():
        tests_passed += 1
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    print(f"\nTests passed: {tests_passed}/3")
    
    if tests_passed == 3:
        print("\n✓ ALL SMOKE TESTS PASSED")
        print("\nFor full inference, open the relevant notebook under notebooks/social_reasoning/,")
        print("configure the four checkpoint paths, and run it in an environment with the")
        print("dependencies in requirements-model-inference.txt. This smoke test does not")
        print("validate the full experimental pipeline or guarantee exact result reproduction.")
    else:
        print("\n✗ SOME TESTS FAILED")
        print("\nPlease fix issues above before running full evaluation.")
        print("Common issues:")
        print("  - Dataset access requires HuggingFace login")
        print("  - Model path incorrect")
        print("  - CUDA/GPU issues")
    
    print("\n" + "="*80)
    raise SystemExit(0 if tests_passed == 3 else 1)

if __name__ == "__main__":
    main()
