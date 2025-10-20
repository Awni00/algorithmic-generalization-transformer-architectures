# Blog Post: Recursive Thinking From Within

This folder contains a self-contained blog post about our paper "Recursive Thinking From Within: Unlocking Out-of-Distribution Generalization in Transformers via Latent Space Reasoning".

## Contents

- `blog_post.md`: The main blog post in Markdown format
- `figures/`: All figures referenced in the blog post (PDF format)

## Viewing the Blog Post

The blog post is written in Markdown and can be viewed:
1. Directly on GitHub
2. In any Markdown viewer
3. Converted to HTML using tools like `pandoc`:
   ```bash
   pandoc blog_post.md -o blog_post.html
   ```

## Figure Formats

Figures are provided in PDF format. For web publishing, you may want to convert them to PNG:
```bash
# Using ImageMagick (if installed)
for file in figures/*.pdf; do
  convert -density 150 "$file" "${file%.pdf}.png"
done
```

## Key Figures Included

### Mechanism Diagrams
- `mech1.pdf` - Recurrence & Adaptive Computation
- `mech2.pdf` - Algorithmic Supervision
- `mech3.pdf` - Anchored Discrete Latent Space
- `mech4.pdf` - Error Correction

### Mechanistic Interpretability Analysis
- `L0_head_allocation.pdf` - First layer attention head specialization
- `L0_var_*_relative_variance_heatmap.pdf` - Variable-specific attention patterns
- `cosine_similarity_heatmap.pdf` - Value embedding structure
- `fft_histograms_*.pdf` - Frequency domain analysis of MLP computation
- `l1_attention_head_statistics_*.pdf` - Second layer attention analysis

### Error Analysis
- `err_*.pdf` - Various error analysis visualizations
- `l2_relative_error.pdf` - MLP modification analysis

## Citation

If you use this work, please cite:
```bibtex
@article{altabaa2024recursive,
  title={Recursive Thinking From Within: Unlocking Out-of-Distribution Generalization in Transformers via Latent Space Reasoning},
  author={Altabaa, Awni and Chen, Siyu and Lafferty, John and Yang, Zhuoran},
  journal={arXiv preprint},
  year={2024}
}
```