<div align="center">

# Unlocking Out-of-Distribution Generalization in Transformers via Recursive Latent Space Reasoning

**Awni Altabaa** · **Siyu Chen** · **John Lafferty** · **Zhuoran Yang**

*Enhancing Transformer architectures with recursive latent space reasoning mechanisms for robust algorithmic generalization*

<!-- TODO: add arxiv link -->

</div>

---

## 💡 Abstract

Systematic, compositional generalization beyond the training distribution remains a core challenge in machine learning—and a critical bottleneck for the emergent reasoning abilities of modern language models.

This work investigates out-of-distribution (OOD) generalization in Transformer networks using a GSM8K-style modular arithmetic on computational graphs task as a testbed.

<!-- 
### ⭐ Key Contributions
-->
We introduce and explore **four architectural mechanisms** aimed at enhancing OOD generalization:

1. **🔄 Input-adaptive recurrence** - Recurrent architecture that scales computation through input-adaptive recurrence.
2. **📚 Algorithmic supervision** - Structured learning objectives that encode algorithmic knowledge  
3. **⚓ Anchored latent representations** - Discrete bottlenecks for stable feature learning
4. **🔧 Explicit error-correction mechanism** - Built-in self-correction capabilities
<!--
### 📊 Results
-->
Collectively, these mechanisms yield an architectural approach for **native and scalable latent space reasoning** in Transformer networks with robust algorithmic generalization capabilities. We complement these empirical results with a detailed **mechanistic interpretability analysis** that reveals how these mechanisms give rise to robust OOD generalization abilities.

---
<!-- 
## 🗂️ Repository Structure

```
📁 algorithmic-generalization-transformer-architectures/
├── 📄 LICENSE                          # Project license
├── 📄 README.md                        # This file
├── 📁 experiments/                     # Main experimental code
│   ├── 🐍 dag_generator.py            # Computational graph generation
│   ├── 🐍 generate_data.py            # Data generation utilities
│   ├── 🐍 model.py                    # Core model implementations
│   ├── 📄 readme.md                   # Detailed experiment instructions
│   ├── 🐍 tokenizers.py               # Custom tokenization methods
│   ├── 🐍 train.py                    # Main training script
│   ├── 🐍 utils.py                    # Utility functions
│   ├── 📁 baselines/                  # Baseline model implementations
│   │   ├── 🐍 baseline_cot_train.py   # Chain-of-thought baseline training
│   │   ├── 🐍 baseline_models.py      # Standard baseline architectures
│   │   ├── 🐍 baseline_train.py       # General baseline training
│   │   └── 🐍 generate_cot_data.py    # CoT data generation
│   ├── 📁 checkpoints/                # Saved model checkpoints
│   │   └── 📁 demo_group-demo_run/    # Example checkpoint
│   ├── 📁 configs/                    # Runtime configuration files
│   ├── 📁 data/                       # Generated datasets
│   │   └── 📁 Tr32Test128-ADD/        # Example dataset (32→128 nodes, ADD operations)
│   │       ├── 🗃️ train_data.pt       # Training dataset
│   │       ├── 🗃️ val_data.pt         # Validation dataset
│   │       └── 🔧 tokenizer.pickle    # Associated tokenizer
│   ├── 📁 evaluation/                 # Model evaluation tools
│   │   ├── 🐍 eval_baseline_model.py  # Baseline model evaluation
│   │   ├── 🐍 eval_cot_baseline_model.py # CoT baseline evaluation
│   │   ├── 🐍 eval_model.py           # Main model evaluation
│   │   ├── 🐍 eval_nointerm_model.py  # No-intermediate evaluation
│   │   ├── 📓 generate_cot_depth_generalization_datasets.ipynb # CoT dataset notebook
│   │   ├── 🐍 generate_cot_depth_generalization_datasets.py # CoT dataset script
│   │   ├── 🐍 generate_depth_generalization_datasets.py # Main dataset generation
│   │   ├── 🐍 metric_utils.py         # Evaluation metrics
│   │   ├── 📁 results/                # Evaluation results
│   │   └── 📁 val_datasets/           # Generated evaluation datasets
│   │       └── 📁 Tr32Test128/        # Example evaluation data
│   ├── 📁 example_configs/            # Example configuration files
│   │   ├── 📁 CoT/                    # Chain-of-Thought configurations
│   │   │   ├── ⚙️ data_config.yaml
│   │   │   ├── ⚙️ model_config.yaml
│   │   │   └── ⚙️ train_config.yaml
│   │   ├── 📁 feedforward_or_recurrent/ # Baseline configurations
│   │   │   ├── ⚙️ data_config.yaml
│   │   │   ├── ⚙️ model_config.yaml
│   │   │   └── ⚙️ train_config.yaml
│   │   └── 📁 our_method/             # Our method configurations
│   │       ├── ⚙️ data_config.yaml
│   │       ├── ⚙️ model_config.yaml
│   │       └── ⚙️ train_config.yaml
│   ├── 📁 lightning_logs/             # Training logs
│   └── 📁 model_interp/               # Model interpretability analysis
│       ├── 🐍 conduct_controlled_exp.py # Controlled experiments
│       ├── 📓 transformer_model_interpretation_documented.ipynb # Analysis notebook
│       ├── 📁 base_config-Nodes32-ADD/ # Interpretation configuration
│       │   ├── ⚙️ data_config.yaml
│       │   ├── ⚙️ model_config.yaml
│       │   └── ⚙️ train_config.yaml
│       ├── 📁 demo_checkpoint/         # Pre-trained model checkpoint
│       │   └── 💾 last.ckpt
│       └── 📁 figures/                 # Visualization outputs
│           ├── 🌐 L0_head_view.html
│           └── 🌐 L0_model_view.html
└── 📁 Simtransformer/                 # Simulation framework
    ├── 📄 README.md                   # Framework documentation
    ├── 📁 configurations/             # Global configuration templates
    │   ├── ⚙️ data_config_default.yaml
    │   ├── ⚙️ model_config_default.yaml
    │   ├── ⚙️ probe_config_default.yaml
    │   └── ⚙️ train_config_default.yaml
    └── 📁 simtransformer/             # Core framework code
        ├── 🐍 manager.py              # Experiment management
        ├── 🐍 model_bank.py           # Model registry
        ├── 🐍 model_base.py           # Base model classes
        ├── 🐍 module_base.py          # Base module classes
        ├── 📄 README.md               # Framework details
        ├── 🐍 utils.py                # Framework utilities
        └── 📁 configurations/         # Local configurations
            ├── ⚙️ data_config_default.yaml
            ├── ⚙️ model_config_default.yaml
            ├── ⚙️ probe_config_default.yaml
            └── ⚙️ train_config_default.yaml
``` -->

### 📂 Organization of Codebase

- **`experiments/`** - Contains the main experimental code for training and evaluating models with the proposed architectural mechanisms
- **`experiments/baselines/`** - Baseline implementations for comparison including chain-of-thought models and standard transformers
- **`experiments/evaluation/`** - Code for evaluating the algorithmic generalization capabilities of different methods.
- **`experiments/model_interp/`** - Mechanistic interpretability analysis tools and visualizations
- **`Simtransformer/`** - Helper framework implementing Transformer modules and related utilities

See [`experiments/readme.md`](experiments/readme.md) for instructions on how to reproduce the experiments in the paper.

<!-- TODO -->
<!-- ## Citation

```bibtex
@article{altabaa2025unlocking,
  title = {Unlocking Out-of-Distribution Generalization in Transformers via Recursive Latent Space Reasoning},
  author = {Altabaa, Awni and Chen, Siyu and Yang, Zhuoran and Lafferty, John},
  year = {2025},
  journal = {arXiv preprint arxiv:[...]}
}
``` -->
