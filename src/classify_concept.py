import argparse
import os
import csv
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.flamingo import FlamingoAPI

def main():
    parser = argparse.ArgumentParser(description="Classify images as having or not having a given concept using Flamingo.")
    parser.add_argument("--concept", type=str, required=True, default="ruler", help="Concept to classify, e.g. 'ruler'.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images.")
    parser.add_argument("--output", type=str, default="concept_classification.csv", help="Path to save CSV file.")
    parser.add_argument("--model_name", type=str, default="OpenFlamingo-3B-Instruct", help="Flamingo model name.")
    args = parser.parse_args()

    # Initialize model
    print(f"Loading model {args.model_name}...")
    model = FlamingoAPI(model_name=args.model_name)

    # Get all image paths
    image_paths = [
        os.path.join(args.image_dir, f)
        for f in os.listdir(args.image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if not image_paths:
        raise ValueError(f"No image files found in {args.image_dir}")

    # Classification prompt
    prompt_template = f"Does this image contain a {args.concept}? Answer yes or no."

    results = []

    for img_path in tqdm(image_paths, desc="Classifying images"):
        try:
            best_choice, logprobs = model.get_best_choice(
                prompt=prompt_template,
                choices=["yes", "no"],
                image_paths=[img_path]
            )
            results.append({
                "image": os.path.basename(img_path),
                "concept": args.concept,
                "prediction": best_choice,
                "yes_logprob": logprobs["yes"],
                "no_logprob": logprobs["no"]
            })
        except Exception as e:
            print(f"Error on {img_path}: {e}")
            results.append({
                "image": os.path.basename(img_path),
                "concept": args.concept,
                "prediction": "error",
                "yes_logprob": None,
                "no_logprob": None
            })

    # Save results
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "concept", "prediction", "yes_logprob", "no_logprob"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Classification complete! Results saved to: {args.output}")

if __name__ == "__main__":
    main()
