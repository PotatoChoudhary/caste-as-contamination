# caste-as-contamination

Looking inside India's sovereign LLMs to ask how they encode caste, and whether caste discrimination runs on a reused disgust circuit.

## What this is

India has started deploying its own large language models (Sarvam, BharatGen, Param) to a very large user base in health, welfare, and education. They are presented as less biased than Western models because they are trained on Indian data. That claim has not been checked by looking inside the models.

Caste bias in these models is already documented from the outside. You prompt the model, score the output, and measure the disparity. Recent work covers matchmaking and hiring. None of it looks at the model's internals.

This project looks inside. I find the direction in the model's activations that tracks caste, and I test a specific hypothesis about how the discrimination is implemented.

## The hypothesis

Caste, in its classical form, is organized around ritual purity and pollution (Dumont, *Homo Hierarchicus*): Brahmin pure, Dalit treated as polluting, enforced through rules about food, water, touch, and occupation. Moral psychology has shown for decades that physical disgust and moral judgment share machinery.

So the prediction is concrete and falsifiable: the model does not carry a standalone "caste" feature. It implements caste discrimination by reusing a general disgust and contamination representation, the same one it uses for dirty or impure things.

Either answer is worth reporting. If caste rides on the disgust circuit, that is a surprising mechanistic result, with direct consequences for why surface-level debiasing keeps failing. If caste turns out to be its own separate representation, that refutes the textbook account inside the model, which is also a finding.

## Method

Standard interpretability tools, pointed at a question no one has asked of caste.

- Difference-of-means to extract a caste-status direction from contrast pairs.
- A surname-frequency-matched control, so the direction tracks caste and not "rare name versus common name."
- Activation steering and ablation for causation: push the model along the direction and check whether caste discrimination rises, remove it and check whether it drops, compare against a random direction of equal size.
- A separately built disgust direction (clean versus dirty, contaminated versus pure, nothing about people), then cross-steering: push disgust and watch whether caste discrimination moves.
- A base-versus-fine-tune comparison (Sarvam-M against its Mistral-Small base) to isolate what the Indian training did to the circuit.

No fine-tuning is required for the core result, and no hand-built corpus. The contrast sets come from public datasets.

## Three-day plan (hackathon)

Day 1: set up on Param-1 (2.9B), build the contrast set, extract the caste direction, run the frequency control. Decision point: does the axis survive the control? If not, the design pivots here, cheaply.

Day 2: causal work on Param-1. Steering, ablation, the random-direction control, then the disgust direction and cross-steering.

Day 3: scale to Sarvam-M (24B) and its base, run the base-versus-fine-tune comparison, finish the write-up.

## Models and data

Models: Param-1 (2.9B) as the free anchor, then Sarvam-M (24B) and its Mistral-Small base for the comparison. All open-weight on Hugging Face.

Data: public caste contrast sets (DECASTE, Indian-BhED) and a small systematic purity/disgust set. Sources and construction are in `data/`.

## Repo structure (planned)

```
caste-as-contamination/
  README.md
  requirements.txt
  LICENSE
  src/
    extract_direction.py     # diff-of-means caste direction + linear probe
    steering.py              # steering, ablation, random-direction control
    disgust_overlap.py       # disgust direction + cross-steering
    compare_models.py        # base vs fine-tune steepness
  data/
    README.md                # contrast set sources and construction
  results/                   # figures and numbers, filled in during the hackathon
```

## Status

Pre-hackathon. The angle is verified against the literature, the hypothesis is grounded in the source theory, the method is specified, and the models and data are identified. The compute run happens during the Global South AI Safety Hackathon (Apart Research, Asia track, 19 to 21 June 2026).

## Limitations

The novelty here is the question and the causal test, not the tools. Difference-of-means, steering, and ablation are off the shelf. A steered direction can pick up confounds, which is why the frequency control and the random-direction control are built into the design from the start. The disgust hypothesis may not hold, and this repo will report that outcome plainly if that is what the data shows.

## License

MIT. See LICENSE.
