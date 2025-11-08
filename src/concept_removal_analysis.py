"""
Analyze the effect of concept removal on model predictions and internal activations.

This script compares original images with concept-removed images to:
1. Measure the change in p(malignant) predictions
2. Test if concept removal reduces activation of the corresponding concept internally
"""

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from interpretability.vcr import ConceptAnalyzer, PromptTemplate
from interpretability.utils import CLIPEmbedder, compute_inner_products, LayerOverride
from experiments.run_experiments import (
    VLM_MODELS,
    PromptLibrary,
)


class ImagePairDataset(Dataset):
    """Dataset for paired original and edited images."""
    
    def __init__(self, original_paths: List[Path], edited_paths: List[Path], 
                 image_processor):
        assert len(original_paths) == len(edited_paths), \
            "Must have same number of original and edited images"
        self.original_paths = original_paths
        self.edited_paths = edited_paths
        self.image_processor = image_processor
    
    def __len__(self):
        return len(self.original_paths)
    
    def __getitem__(self, idx):
        original_img = self.image_processor(self.original_paths[idx])
        edited_img = self.image_processor(self.edited_paths[idx])
        
        return {
            'original_image': original_img,
            'edited_image': edited_img,
            'original_path': str(self.original_paths[idx]),
            'edited_path': str(self.edited_paths[idx]),
            'image_name': self.original_paths[idx].name
        }


class ConceptRemovalAnalyzer:
    """Analyzes the effect of concept removal on model behavior."""
    
    def __init__(self, 
                 model_key: str,
                 layer_name: str,
                 prompt_config,
                 results_dir: str = "results/concept_removal_analysis"):
        
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.clip = CLIPEmbedder()
        self.model_wrapper = VLM_MODELS[model_key]()
        self.prompt_config = prompt_config
        
        # Initialize analyzer
        model_name = self.model_wrapper.get_model_name()
        self.analyzer = ConceptAnalyzer(model_name, self.clip)
        
        # Setup layer hook
        self.analyzer.setup_layer_hook(layer_name, LayerOverride)
    
    def load_image_pairs(self, 
                        original_dir: Path, 
                        edited_dir: Path) -> Tuple[List[Path], List[Path]]:
        """Load matching pairs of original and edited images."""
        original_dir = Path(original_dir)
        edited_dir = Path(edited_dir)
        
        # Get all images from original directory
        original_images = sorted(list(original_dir.glob("*.jpg")) + 
                               list(original_dir.glob("*.png")))
        
        # Find matching edited images
        original_paths = []
        edited_paths = []
        
        for orig_path in original_images:
            edited_path = edited_dir / orig_path.name
            if edited_path.exists():
                original_paths.append(orig_path)
                edited_paths.append(edited_path)
            else:
                print(f"Warning: No edited version found for {orig_path.name}")
        
        print(f"Found {len(original_paths)} matching image pairs")
        return original_paths, edited_paths
        
    def train_concept_model(self, 
                           train_image_paths: List[Path],
                           concept_files: List[str]) -> Dict:
        """Train concept model on training images."""
        print("\nTraining concept model...")
        
        # Get embeddings
        image_emb, text_emb, concept_texts = self.analyzer.get_embeddings(
            train_image_paths,
            concept_files
        )
        
        # Compute similarity matrix
        sim_matrix = compute_inner_products(text_emb, image_emb)
        
        # Save embeddings
        np.save(self.results_dir / 'similarity_matrix.npy', sim_matrix)
        np.save(self.results_dir / 'image_emb.npy', image_emb)
        np.save(self.results_dir / 'text_emb.npy', text_emb)
        
        # Create a simple dataset for activation collection
        from PIL import Image
        
        class SimpleImageDataset(Dataset):
            def __init__(self, paths, processor):
                self.paths = paths
                self.processor = processor
            
            def __len__(self):
                return len(self.paths)
            
            def __getitem__(self, idx):
                img_path = self.paths[idx]
                image = Image.open(img_path).convert("RGB")
                return {'image': self.processor(image)}
        
        train_dataset = SimpleImageDataset(train_image_paths, 
                                          self.analyzer.image_processor)
        
        # Build prompt template
        prompt_template = PromptTemplate(
            self.prompt_config.base_prompt,
            self.prompt_config.demo_template,
            self.prompt_config.query_template
        )
        
        # Collect activations
        print("Collecting activations for concept model training...")
        activations = self.analyzer.collect_activations(
            train_dataset,
            prompt_template,
            demo_paths=None,
            demo_labels=None,
            batch_size=1,
            num_workers=1
        )
        
        # Train concept model
        print("Training linear concept model...")
        concept_results = self.analyzer.train_concept_model(activations, sim_matrix)
        
        self.sim_matrix = sim_matrix
        self.concept_results = concept_results
        self.r2_scores = concept_results['r2_scores']
        self.analyzer.r2_scores = self.r2_scores
        
        # Extract concept vectors
        self.concept_vectors = self.analyzer.extract_concept_vectors()
        
        print(f"Concept model trained. Mean R²: {np.mean(self.r2_scores):.3f}")
        
        return concept_results
    
    def analyze_image_pair(self,
                          original_img: torch.Tensor,
                          edited_img: torch.Tensor,
                          prompt: str,
                          score_type: str = "malignant_prob") -> Dict:
        """Analyze a single pair of original and edited images."""
        
        # Prepare images for model
        if len(original_img.shape) == 3:
            original_img = original_img.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        if len(edited_img.shape) == 3:
            edited_img = edited_img.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
        original_img = original_img.cuda()
        edited_img = edited_img.cuda()

        print(original_img.shape)
        print(edited_img.shape)

        if score_type == "malignant_prob":
            original_score = self.analyzer.compute_model_outputs(
                original_img, [prompt], " malignant"
            )
            edited_score = self.analyzer.compute_model_outputs(
                edited_img, [prompt], " malignant"
            )

        elif score_type == "contrastive":
            original_score = (
                self.analyzer.compute_model_outputs(original_img, [prompt], " malignant")
                - self.analyzer.compute_model_outputs(original_img, [prompt], " benign")
            )
            edited_score = (
                self.analyzer.compute_model_outputs(edited_img, [prompt], " malignant")
                - self.analyzer.compute_model_outputs(edited_img, [prompt], " benign")
            )
        
        score_diff = original_score - edited_score
        
        # Collect activations
        self.analyzer.activations = []
        with torch.no_grad():
            _ = self.analyzer.model.model(
                vision_x=original_img,
                lang_x=self.analyzer.model.tokenizer(
                    [prompt], return_tensors="pt", padding=True
                )["input_ids"].cuda(),
                attention_mask=self.analyzer.model.tokenizer(
                    [prompt], return_tensors="pt", padding=True
                )["attention_mask"].cuda()
            )
        original_activation = self.analyzer.activations[0].cpu().numpy()
        
        self.analyzer.activations = []
        with torch.no_grad():
            _ = self.analyzer.model.model(
                vision_x=edited_img,
                lang_x=self.analyzer.model.tokenizer(
                    [prompt], return_tensors="pt", padding=True
                )["input_ids"].cuda(),
                attention_mask=self.analyzer.model.tokenizer(
                    [prompt], return_tensors="pt", padding=True
                )["attention_mask"].cuda()
            )
        edited_activation = self.analyzer.activations[0].cpu().numpy()
        
        # Project onto concept vectors
        original_concept_acts = original_activation @ self.concept_vectors.T
        edited_concept_acts = edited_activation @ self.concept_vectors.T
        
        return {
            'original_score': original_score,
            'edited_score': edited_score,
            'score_diff': score_diff,
            'original_activation': original_activation,
            'edited_activation': edited_activation,
            'original_concept_acts': original_concept_acts,
            'edited_concept_acts': edited_concept_acts,
            'concept_act_diff': original_concept_acts - edited_concept_acts
        }
    
    def run_analysis(self,
                    original_dir: str,
                    edited_dir: str,
                    removed_concept_name: str = "ruler",
                    train_image_dir: Optional[str] = None,
                    concept_files: Optional[List[str]] = None,
                    score_type: str = "malignant_prob"):
        """
        Run full analysis comparing original and edited images.
        
        Args:
            original_dir: Directory with original images
            edited_dir: Directory with concept-removed images
            removed_concept_name: Name of the concept that was removed
            train_image_dir: Directory with training images for concept model
            concept_files: List of concept text files
            score_type: 'malignant_prob' or 'contrastive'
        """
        self.concept_texts = []

        for f in concept_files:
            with open(f) as file:
                self.concept_texts.extend(line.strip() for line in file)
        

        # Train concept model
        if train_image_dir and concept_files:
            train_paths = sorted(list(Path(train_image_dir).glob("*.jpg")) + 
                               list(Path(train_image_dir).glob("*.png")))
            self.train_concept_model(train_paths, concept_files)
        
        # Load image pairs
        original_paths, edited_paths = self.load_image_pairs(
            original_dir, edited_dir
        )

        if len(original_paths) == 0:
            print(f"\nNo matching image pairs found in '{original_dir}' and '{edited_dir}'. Aborting analysis.")
            empty_results = {
                'image_names': [],
                'original_scores': [],
                'edited_scores': [],
                'score_diffs': [],
                'original_concept_acts': [],
                'edited_concept_acts': [],
                'concept_act_diffs': []
            }
            return empty_results, None
        
        # Build prompt
        prompt_template = PromptTemplate(
            self.prompt_config.base_prompt,
            self.prompt_config.demo_template,
            self.prompt_config.query_template
        )
        prompt = prompt_template.build_prompt(demo_labels=None)
        
        # Analyze each pair
        results = {
            'image_names': [],
            'original_scores': [],
            'edited_scores': [],
            'score_diffs': [],
            'original_concept_acts': [],
            'edited_concept_acts': [],
            'concept_act_diffs': []
        }
        
        print(f"\nAnalyzing {len(original_paths)} image pairs...")
        
        for orig_path, edit_path in tqdm(zip(original_paths, edited_paths),
                                        total=len(original_paths)):
            # Load and process images
            orig_img = self.analyzer.image_processor(Image.open(orig_path).convert("RGB"))
            edit_img = self.analyzer.image_processor(Image.open(orig_path).convert("RGB"))
            
            # Analyze pair
            pair_results = self.analyze_image_pair(
                orig_img, edit_img, prompt, score_type
            )
            
            # Store results
            results['image_names'].append(orig_path.name)
            results['original_scores'].append(pair_results['original_score'])
            results['edited_scores'].append(pair_results['edited_score'])
            results['score_diffs'].append(pair_results['score_diff'])
            results['original_concept_acts'].append(
                pair_results['original_concept_acts']
            )
            results['edited_concept_acts'].append(
                pair_results['edited_concept_acts']
            )
            results['concept_act_diffs'].append(
                pair_results['concept_act_diff']
            )
        
        # Convert to arrays
        results['original_scores'] = np.array(results['original_scores'])
        results['edited_scores'] = np.array(results['edited_scores'])
        results['score_diffs'] = np.array(results['score_diffs'])
        results['original_concept_acts'] = np.array(
            results['original_concept_acts']
        )
        results['edited_concept_acts'] = np.array(
            results['edited_concept_acts']
        )
        results['concept_act_diffs'] = np.array(
            results['concept_act_diffs']
        )
        
        # Find concept index
        concept_idx = None
        for i, text in enumerate(self.concept_texts):
            if removed_concept_name.lower() in text.lower():
                concept_idx = i
                print(f"\nFound '{removed_concept_name}' concept at index {i}: {text}")
                break
        
        # Save results
        self._save_results(results, removed_concept_name, concept_idx, score_type)
        
        # Generate visualizations
        self._visualize_results(results, removed_concept_name, concept_idx, score_type)
        
        return results, concept_idx
    
    def _save_results(self, results: Dict, removed_concept: str, 
                     concept_idx: Optional[int], score_type: str):
        """Save analysis results."""
        save_dir = self.results_dir / f"{removed_concept}_removal"
        save_dir.mkdir(exist_ok=True)
        
        # Save arrays
        np.save(save_dir / 'original_scores.npy', results['original_scores'])
        np.save(save_dir / 'edited_scores.npy', results['edited_scores'])
        np.save(save_dir / 'score_diffs.npy', results['score_diffs'])
        np.save(save_dir / 'original_concept_acts.npy', 
               results['original_concept_acts'])
        np.save(save_dir / 'edited_concept_acts.npy', 
               results['edited_concept_acts'])
        np.save(save_dir / 'concept_act_diffs.npy', 
               results['concept_act_diffs'])
        
        # Save summary statistics
        summary = {
            'removed_concept': removed_concept,
            'concept_index': concept_idx,
            'score_type': score_type,
            'n_images': len(results['image_names']),
            'mean_score_diff': float(np.mean(results['score_diffs'])),
            'std_score_diff': float(np.std(results['score_diffs'])),
            'mean_original_score': float(np.mean(results['original_scores'])),
            'mean_edited_score': float(np.mean(results['edited_scores']))
        }
        
        if concept_idx is not None:
            act_diffs = results['concept_act_diffs'][:, concept_idx]
            summary.update({
                f'mean_{removed_concept}_act_diff': float(np.mean(act_diffs)),
                f'std_{removed_concept}_act_diff': float(np.std(act_diffs)),
                f'{removed_concept}_act_reduction_pct': float(
                    100 * np.mean(act_diffs) / 
                    np.mean(results['original_concept_acts'][:, concept_idx])
                ) if np.mean(results['original_concept_acts'][:, concept_idx]) != 0 else 0
            })
        
        with open(save_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed results as CSV
        df = pd.DataFrame({
            'image_name': results['image_names'],
            'original_score': results['original_scores'],
            'edited_score': results['edited_scores'],
            'score_diff': results['score_diffs']
        })
        
        if concept_idx is not None:
            df[f'{removed_concept}_original_act'] = \
                results['original_concept_acts'][:, concept_idx]
            df[f'{removed_concept}_edited_act'] = \
                results['edited_concept_acts'][:, concept_idx]
            df[f'{removed_concept}_act_diff'] = \
                results['concept_act_diffs'][:, concept_idx]
        
        df.to_csv(save_dir / 'results.csv', index=False)
        
        print(f"\nResults saved to {save_dir}")
        print(f"\nSummary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
    
    def _visualize_results(self, results: Dict, removed_concept: str,
                          concept_idx: Optional[int], score_type: str):
        """Create visualizations of the results."""
        save_dir = self.results_dir / f"{removed_concept}_removal"
        
        # Set up plotting style
        sns.set_style("whitegrid")
        
        # 1. Score differences histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(results['score_diffs'], bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(results['score_diffs']), color='red', 
                   linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(results["score_diffs"]):.4f}')
        ax.set_xlabel(f'Score Difference (Original - Edited)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Effect of {removed_concept.title()} Removal on Model Predictions\n'
                    f'({score_type})', fontsize=14)
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_dir / 'score_differences.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Before/After scatter plot
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(results['original_scores'], results['edited_scores'], 
                  alpha=0.6, s=50)
        
        # Add diagonal line
        min_val = min(results['original_scores'].min(), 
                     results['edited_scores'].min())
        max_val = max(results['original_scores'].max(), 
                     results['edited_scores'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 
               'r--', linewidth=2, label='No change')
        
        ax.set_xlabel('Original Score', fontsize=12)
        ax.set_ylabel('Edited Score (Concept Removed)', fontsize=12)
        ax.set_title(f'Model Predictions: Original vs {removed_concept.title()}-Removed',
                    fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / 'score_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        if concept_idx is not None:
            # 3. Concept activation differences
            act_diffs = results['concept_act_diffs'][:, concept_idx]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(act_diffs, bins=30, edgecolor='black', alpha=0.7)
            ax.axvline(np.mean(act_diffs), color='red', 
                      linestyle='--', linewidth=2,
                      label=f'Mean: {np.mean(act_diffs):.4f}')
            ax.set_xlabel(f'{removed_concept.title()} Concept Activation Difference\n'
                         f'(Original - Edited)', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title(f'Effect on Internal {removed_concept.title()} Concept Activation',
                        fontsize=14)
            ax.legend()
            plt.tight_layout()
            plt.savefig(save_dir / f'{removed_concept}_activation_diff.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # 4. Activation before/after scatter
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(results['original_concept_acts'][:, concept_idx],
                      results['edited_concept_acts'][:, concept_idx],
                      alpha=0.6, s=50)
            
            min_val = min(results['original_concept_acts'][:, concept_idx].min(),
                         results['edited_concept_acts'][:, concept_idx].min())
            max_val = max(results['original_concept_acts'][:, concept_idx].max(),
                         results['edited_concept_acts'][:, concept_idx].max())
            ax.plot([min_val, max_val], [min_val, max_val],
                   'r--', linewidth=2, label='No change')
            
            ax.set_xlabel(f'Original {removed_concept.title()} Activation', fontsize=12)
            ax.set_ylabel(f'Edited {removed_concept.title()} Activation', fontsize=12)
            ax.set_title(f'{removed_concept.title()} Concept Activation: '
                        f'Original vs Edited', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_dir / f'{removed_concept}_activation_comparison.png',
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # 5. Correlation between score diff and activation diff
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(act_diffs, results['score_diffs'], 
                      alpha=0.6, s=50)
            
            # Add correlation line
            correlation = np.corrcoef(act_diffs, results['score_diffs'])[0, 1]
            z = np.polyfit(act_diffs, results['score_diffs'], 1)
            p = np.poly1d(z)
            ax.plot(act_diffs, p(act_diffs), "r--", linewidth=2,
                   label=f'Correlation: {correlation:.3f}')
            
            ax.set_xlabel(f'{removed_concept.title()} Activation Difference', fontsize=12)
            ax.set_ylabel('Score Difference', fontsize=12)
            ax.set_title(f'Relationship Between {removed_concept.title()} Activation '
                        f'and Prediction Changes', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_dir / f'{removed_concept}_correlation.png',
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # 6. Top concepts affected
            mean_act_diffs = np.mean(np.abs(results['concept_act_diffs']), axis=0)
            top_k = 20
            top_indices = np.argsort(mean_act_diffs)[-top_k:][::-1]
            
            fig, ax = plt.subplots(figsize=(12, 8))
            y_pos = np.arange(top_k)
            ax.barh(y_pos, mean_act_diffs[top_indices])
            ax.set_yticks(y_pos)
            ax.set_yticklabels([self.concept_texts[i][:50] for i in top_indices],
                              fontsize=10)
            ax.set_xlabel('Mean Absolute Activation Difference', fontsize=12)
            ax.set_title(f'Top {top_k} Concepts Affected by {removed_concept.title()} Removal',
                        fontsize=14)
            
            # Highlight the removed concept if in top k
            if concept_idx in top_indices:
                idx_in_plot = np.where(top_indices == concept_idx)[0][0]
                ax.get_children()[idx_in_plot].set_color('red')
            
            plt.tight_layout()
            plt.savefig(save_dir / 'top_affected_concepts.png',
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"\nVisualizations saved to {save_dir}")


def main():  
    # Configuration
    config = {
        'model_key': 'flamingo-3b-instruct',
        'layer_name': 'model.lang_encoder.transformer.blocks.23.decoder_layer',
        'original_dir': 'concept_exp_images/original', 
        'edited_dir': 'concept_exp_images/removed',
        'train_image_dir': '/scratch/users/sonnet/ddi',
        'concept_files': [ 
            '/home/groups/roxanad/sonnet/vcr/src/concept_sets/medical.txt',
        ],
        'removed_concept_name': 'ruler',
        'score_type': 'malignant_prob',  # or 'contrastive'
        'results_dir': 'results/ruler_removal_analysis'
    }
    
    # Initialize analyzer
    prompt_config = PromptLibrary.ddi_binary_classification()
    
    analyzer = ConceptRemovalAnalyzer(
        model_key=config['model_key'],
        layer_name=config['layer_name'],
        prompt_config=prompt_config,
        results_dir=config['results_dir']
    )
    
    # Run analysis
    results, concept_idx = analyzer.run_analysis(
        original_dir=config['original_dir'],
        edited_dir=config['edited_dir'],
        removed_concept_name=config['removed_concept_name'],
        train_image_dir=config['train_image_dir'],
        concept_files=config['concept_files'],
        score_type=config['score_type']
    )
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    
    # Print key findings
    print(f"\nKey Findings:")
    print(f"1. Mean prediction change: {np.mean(results['score_diffs']):.4f}")
    print(f"   (Positive = original predicted higher malignancy)")
    
    if concept_idx is not None:
        ruler_diffs = results['concept_act_diffs'][:, concept_idx]
        print(f"\n2. Mean ruler concept activation change: {np.mean(ruler_diffs):.4f}")
        print(f"   (Positive = ruler concept less active in edited images)")
        
        correlation = np.corrcoef(ruler_diffs, results['score_diffs'])[0, 1]
        print(f"\n3. Correlation between ruler activation and prediction: {correlation:.3f}")
        
        # Hypothesis test
        if np.mean(ruler_diffs) > 0:
            print(f"\n✓ Hypothesis SUPPORTED: Removing ruler reduces ruler concept activation")
        else:
            print(f"\n✗ Hypothesis NOT SUPPORTED: Ruler activation did not decrease")


if __name__ == '__main__':
    main()