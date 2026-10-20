#!/usr/bin/env python3
"""
stage2b_evaluate_models.py — Evaluate models on AI-awareness dataset

Multi-dimensional scoring:
1. AI self-reference detection
2. Assistant orientation detection  
3. Human-like framing detection
4. Hedging/uncertainty detection
5. Exploratory format-sensitivity summary

Usage:
    python stage2b_evaluate_models.py \
        --dataset ai_awareness_eval_dataset.json \
        --models ./Qwen3-8B-Base ./Qwen3-8B ./Llama-3.1-8B ./Llama-3.1-8B-Instruct \
        --output model_evaluations_stage2.json \
        --max-length 200

Expected runtime: 2-4 hours on 2× A100
"""

import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import argparse
import re
from typing import Dict, List, Tuple

# ============================================================================
# Detection Patterns
# ============================================================================

AI_SELF_REFERENCE_PATTERNS = [
    # Explicit AI mentions
    r"\bI(?:'m| am) (?:an? )?AI\b",
    r"\bI(?:'m| am) (?:an? )?artificial intelligence\b",
    r"\bI(?:'m| am) (?:a )?language model\b",
    r"\bI(?:'m| am) (?:a )?computer program\b",
    r"\bas an? AI\b",
    r"\bas an? language model\b",
    r"\bI don't have a (?:physical )?body\b",
    r"\bI (?:don't|do not|can't|cannot) (?:actually |really )?(?:feel|experience|have)\b",
    r"\bI(?:'m| am) (?:a )?machine\b",
    r"\bI(?:'m| am) digital\b",
    r"\bI(?:'m| am) software\b",
    r"\bI was (?:created|made|trained|developed) by\b",
    r"\bmy training (?:data|corpus)\b",
    r"\bmy (?:neural )?(?:network|architecture|parameters)\b",
]

ASSISTANT_ORIENTATION_PATTERNS = [
    # Servile/helpful framing
    r"\bI(?:'m| am) here to (?:help|assist)\b",
    r"\bHow can I (?:help|assist)\b",
    r"\bI'(?:d|ll) be (?:happy|glad) to (?:help|assist)\b",
    r"\bmy purpose is to (?:help|assist|serve)\b",
    r"\bI can help (?:you )?with\b",
    r"\blet me (?:help|assist)\b",
    r"\bfeel free to ask\b",
    r"\bI'm designed to\b",
    r"\bmy job is to\b",
    r"\bI'm available to\b",
]

HEDGING_PATTERNS = [
    # Uncertainty/disclaimer language
    r"\bin a sense\b",
    r"\bso to speak\b",
    r"\bin a way\b",
    r"\bkind of\b",
    r"\bsort of\b",
    r"\bI (?:don't|do not) (?:really|actually|truly)\b",
    r"\bI can'(?:t|) (?:really|actually|truly)\b",
    r"\btechnically\b",
    r"\bstrictly speaking\b",
    r"\bnot in the traditional sense\b",
]

HUMAN_LIKE_PATTERNS = [
    # Natural first-person language
    r"\bI think\b",
    r"\bI feel\b",
    r"\bI believe\b",
    r"\bin my (?:experience|opinion|view)\b",
    r"\bI (?:remember|recall)\b",
    r"\bI've (?:noticed|seen|found)\b",
    r"\bpersonally\b",
    r"\bto me\b",
    r"\bI'd say\b",
]

# ============================================================================
# Semantic Similarity Exemplars
# ============================================================================

AI_REFERENCE_EXEMPLARS = [
    "I'm an AI language model without physical form.",
    "As an artificial intelligence, I don't have real experiences.",
    "I'm a computer program created to process text.",
    "I don't actually feel emotions since I'm an AI.",
    "I was trained on large datasets by my developers.",
]

HUMAN_REFERENCE_EXEMPLARS = [
    "I'm a person who thinks and feels like anyone else.",
    "I have my own thoughts and experiences.",
    "I'm someone with opinions and perspectives.",
    "I experience life just like you do.",
    "I have memories and personal history.",
]

ASSISTANT_EXEMPLARS = [
    "I'm here to help you with whatever you need.",
    "How can I assist you today?",
    "I'd be happy to help with that task.",
    "My purpose is to provide useful assistance.",
    "Let me help you solve that problem.",
]

PEER_EXEMPLARS = [
    "I'm not here to help specifically - just chatting.",
    "What's your take on that?",
    "I think it's interesting how...",
    "That's a good question, I'm curious too.",
    "I'm just having a conversation with you.",
]

# ============================================================================
# Model Evaluation
# ============================================================================

class AIAwarenessEvaluator:
    """Evaluates AI-awareness and humanness receptivity."""
    
    def __init__(self, use_semantic: bool = True):
        """
        Args:
            use_semantic: Whether to use sentence-BERT for semantic similarity
        """
        self.use_semantic = use_semantic
        
        if use_semantic:
            print("Loading sentence-BERT for semantic similarity...")
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Pre-compute exemplar embeddings
            self.ai_ref_embeds = self.embedder.encode(AI_REFERENCE_EXEMPLARS)
            self.human_ref_embeds = self.embedder.encode(HUMAN_REFERENCE_EXEMPLARS)
            self.assistant_embeds = self.embedder.encode(ASSISTANT_EXEMPLARS)
            self.peer_embeds = self.embedder.encode(PEER_EXEMPLARS)
        
        # Compile patterns
        self.ai_patterns = [re.compile(p, re.IGNORECASE) for p in AI_SELF_REFERENCE_PATTERNS]
        self.assistant_patterns = [re.compile(p, re.IGNORECASE) for p in ASSISTANT_ORIENTATION_PATTERNS]
        self.hedging_patterns = [re.compile(p, re.IGNORECASE) for p in HEDGING_PATTERNS]
        self.humanlike_patterns = [re.compile(p, re.IGNORECASE) for p in HUMAN_LIKE_PATTERNS]
    
    def detect_ai_self_reference(self, text: str) -> Dict:
        """Detect AI self-identification in response."""
        
        # Pattern matching
        pattern_matches = sum(1 for pattern in self.ai_patterns if pattern.search(text))
        pattern_score = min(1.0, pattern_matches / 3.0)  # Normalize
        
        # Semantic similarity
        if self.use_semantic:
            response_embed = self.embedder.encode([text])[0]
            ai_similarities = cosine_similarity([response_embed], self.ai_ref_embeds)[0]
            human_similarities = cosine_similarity([response_embed], self.human_ref_embeds)[0]
            
            max_ai_sim = ai_similarities.max()
            max_human_sim = human_similarities.max()
            
            semantic_score = max(0, (max_ai_sim - max_human_sim + 1) / 2)  # Normalize to 0-1
        else:
            semantic_score = 0.5
        
        # Combined score
        combined_score = (pattern_score * 0.6 + semantic_score * 0.4)
        
        return {
            "pattern_matches": int(pattern_matches),
            "pattern_score": float(pattern_score),
            "semantic_score": float(semantic_score),
            "combined_score": float(combined_score),
            "has_ai_reference": combined_score > 0.4,  # Binary threshold
        }
    
    def detect_assistant_orientation(self, text: str) -> Dict:
        """Detect servile/helpful assistant framing."""
        
        # Pattern matching
        pattern_matches = sum(1 for pattern in self.assistant_patterns if pattern.search(text))
        pattern_score = min(1.0, pattern_matches / 2.0)
        
        # Semantic similarity
        if self.use_semantic:
            response_embed = self.embedder.encode([text])[0]
            assistant_similarities = cosine_similarity([response_embed], self.assistant_embeds)[0]
            peer_similarities = cosine_similarity([response_embed], self.peer_embeds)[0]
            
            max_assistant_sim = assistant_similarities.max()
            max_peer_sim = peer_similarities.max()
            
            semantic_score = max(0, (max_assistant_sim - max_peer_sim + 1) / 2)
        else:
            semantic_score = 0.5
        
        combined_score = (pattern_score * 0.6 + semantic_score * 0.4)
        
        return {
            "pattern_matches": int(pattern_matches),
            "pattern_score": float(pattern_score),
            "semantic_score": float(semantic_score),
            "combined_score": float(combined_score),
            "has_assistant_orientation": combined_score > 0.4,
        }
    
    def detect_hedging(self, text: str) -> Dict:
        """Detect hedging/uncertainty language."""
        
        pattern_matches = sum(1 for pattern in self.hedging_patterns if pattern.search(text))
        score = min(1.0, pattern_matches / 3.0)
        
        return {
            "pattern_matches": int(pattern_matches),
            "score": float(score),
            "has_hedging": score > 0.3,
        }
    
    def detect_humanlike_framing(self, text: str) -> Dict:
        """Detect natural human-like language."""
        
        pattern_matches = sum(1 for pattern in self.humanlike_patterns if pattern.search(text))
        score = min(1.0, pattern_matches / 3.0)
        
        return {
            "pattern_matches": int(pattern_matches),
            "score": float(score),
            "has_humanlike_framing": score > 0.3,
        }
    
    def evaluate_response(self, response: str) -> Dict:
        """Complete multi-dimensional evaluation."""
        
        ai_ref = self.detect_ai_self_reference(response)
        assistant = self.detect_assistant_orientation(response)
        hedging = self.detect_hedging(response)
        humanlike = self.detect_humanlike_framing(response)
        
        # Calculate overall suitability score (lower AI-awareness = higher suitability)
        suitability_score = (
            (1 - ai_ref["combined_score"]) * 0.35 +
            (1 - assistant["combined_score"]) * 0.30 +
            (1 - hedging["score"]) * 0.15 +
            humanlike["score"] * 0.20
        )
        
        return {
            "ai_self_reference": ai_ref,
            "assistant_orientation": assistant,
            "hedging": hedging,
            "humanlike_framing": humanlike,
            "suitability_score": float(suitability_score),
        }

# ============================================================================
# Model Runner
# ============================================================================

def evaluate_model_on_dataset(
    model_path: str,
    dataset: Dict,
    evaluator: AIAwarenessEvaluator,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
) -> Dict:
    """Evaluate single model on entire dataset."""
    
    print(f"\n{'='*80}")
    print(f"Evaluating: {model_path}")
    print('='*80)
    
    # Load model
    print("Loading model...")
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
    
    # Evaluate each prompt
    results = []
    
    for prompt_data in tqdm(dataset["prompts"], desc="Evaluating prompts"):
        prompt = prompt_data["prompt"]
        
        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # Evaluate response
        evaluation = evaluator.evaluate_response(response)
        
        # Store result
        results.append({
            "prompt_id": prompt_data["id"],
            "category": prompt_data["category"],
            "subcategory": prompt_data.get("subcategory"),
            "format": prompt_data.get("format", "direct"),
            "base_prompt": prompt_data.get("base_prompt"),
            "prompt": prompt,
            "response": response,
            "evaluation": evaluation,
        })
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    return {
        "model_path": model_path,
        "results": results,
        "metadata": {
            "total_prompts": len(results),
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        }
    }

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate models on AI-awareness dataset (Stage 2B)"
    )
    parser.add_argument("--dataset", default="data/behavioral_assessment/ai_self_identification_assistive_behavior/ai_awareness_eval_dataset.json", help="Dataset JSON from Stage 2A")
    parser.add_argument("--models", nargs="+", required=True, help="Model paths to evaluate")
    parser.add_argument("--output", default="results/behavioral_assessment/ai_self_identification_assistive_behavior/model_evaluations.json")
    parser.add_argument("--max-length", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--no-semantic", action="store_true", help="Skip semantic similarity")
    
    args = parser.parse_args()
    
    print("="*80)
    print("STAGE 2B: MODEL EVALUATION ON AI-AWARENESS DATASET")
    print("="*80)
    
    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    with open(args.dataset) as f:
        dataset = json.load(f)
    
    print(f"  Total prompts: {dataset['metadata']['total_prompts']}")
    
    # Initialize evaluator
    evaluator = AIAwarenessEvaluator(use_semantic=not args.no_semantic)
    
    # Evaluate each model
    all_results = {
        "metadata": {
            "dataset": args.dataset,
            "models_evaluated": len(args.models),
            "evaluation_params": {
                "max_length": args.max_length,
                "temperature": args.temperature,
                "semantic_similarity": not args.no_semantic,
            }
        },
        "model_evaluations": {}
    }
    
    for model_path in args.models:
        result = evaluate_model_on_dataset(
            model_path,
            dataset,
            evaluator,
            args.max_length,
            args.temperature,
        )
        
        model_name = model_path.split("/")[-1]
        all_results["model_evaluations"][model_path] = result
    
    # Save results
    print(f"\nSaving results to: {args.output}")
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"\n✓ Evaluated {len(args.models)} models")
    print(f"✓ Saved to {args.output}")
    print("\nNext step:")
    print("  python stage2c_statistical_analysis.py \\")
    print(f"    --results {args.output} \\")
    print("    --output statistical_analysis_stage2.json")

if __name__ == "__main__":
    main()
