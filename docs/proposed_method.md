# Fire Horses proposed method.

## Overall idea

We are proposing an experiment design that combines multiple proven ideas for NER.
The method involves training an ensemble of BART learners using PET and OADA.
The ensemble will then produce a soft labeled training set which we distill into a standard parallel BERT classifier.
We will compare scores agains a similar BERT classifier trained on the original data only and compare performance.

## Ensemble members

We will use BART-base initially and move to BART-large, when a working pipeline is established.

### Ensemble training strategy

The ensemble consist of BART models each with their own augmented training set.

**Ensemble members:**

- Each member has a unique random seed.
- Each member has a distinct wrapper pattern (inspired by PET).
  E.g.
  Sentence: `SEN` Order:`PERM`      *(Original OADA pattern)*
  In this sentence: `SEN` List the entities in the following order: `PERM `
  List all entities in this sentence: `SEN` Order: `PERM`
  Using the order: `PERM`, list the entities in this sentence: `SEN`
- Each member has all OADA permutations in their augmented data set.

The labels are represented by short abbrieviations when added to the text input like ORG, LOC, PER, MISC.

The ouput is evaluated against a strictly formatted syntax.
Example output: "[Mikkel]PER [Oslo]LOC"

Loss function is XE loss and OADA XE loss.
Annealing schedule similar to the original OADA method is applied.

**Ensemble inference**
(Creating soft labels for model distillation)

- The `PERM` pattern is replaced by "left to right" similar to the original OADA method.
- A probability distribution over the labels (including the "O" label) is created for each soft label.
  The probability distribution is created be adding the output logics from each member, and applying softmax with T = 2.
  Each word now has a soft label consisting of a vector of length K (including the "O" class).
- The ensemble labels the full training set.

**Soft label postprocessing**

- A strategy to filter out high perplexity labels might be applied.
- The probability distribution vector label can be converted to an argmax hard label for comparison

## Distillation

A standard parallel BERT model is trained on the p-distribution vectors.
Maybe another is trained on the hard labels for comparison.

Loss function consider the label of the first subtoken in each word.

**Final scoring**
The BERT model trained on raw few-shot data is compared to the distilled model trained on soft labels.

---

## Other scores to report

* **Standard BERT** vs **BERT distilled from soft labels** vs **BERT distilled from hard labels**
* **F1 score** as a function of **k-value**

## Ablation studies

1. **Standard BERT** (The baseline tagger)
2. **BART + standard generation** (Generative baseline without OADA)
3. **BART + OADA** (Adding the permutations)
4. **BART + PET + OADA** (The full teacher ensemble with distinct natural language wrappers)
5. **Distilled BERT** (The final student model).

## Logging

For

* **Loss** over **Iterations** is logged for each run.
* **F1 score** as a function of **iterations**
* IDEA: OADA XE activation frequency over iterations.
  (Every time OADA XE loss is less than standard XE loss).

---

## Training optimization

Evaluation when used for applying a stopping strategy, quickly becomes the most expensive part of the training.

**Solution:** A mini-val set is created containing ~100 samples. This set is shared accross all models and seeds. The data is unique to MINI-VAL to avoid overfitting, when scored on VAL.

## Hyperparameters

### BART Ensemble model hyper parameters

**Model:** Bart-base -> BART-large

**Ensemble count:**

- 10 members for k=5, 10
- 5 members for k>=20

**PET patterns:** one distinct pattern per member.

**Number of OADA Permutations:**

- 24 for 4-class data,
- 20 random perms for higher class count.

**BART Training Mechanics**

* **Batch Size:** 16 (on RTX 3090).
* **Learning Rate:**. $1⋅10^{−5}$ (heuristics from PET research).
* **Dropout:** Use heuristics from OADA/PET papers.
* **Stopping paradigm:**

  * **Minimum Steps:** 1,000 (Ensures base syntax is learned before evaluation starts).
  * **Maximum Steps:** 5,000 (The "Hard Cutoff" to prevent GPU waste).
  * **Check Interval:** 200 steps (Run MINI-VAL evaluation).
  * **Early Stopping:** Patience = 5 (Stop if no NEW Best F1 on Set B after 5 checks / 1,000 steps).

### BERT Baseline Classifier

The goal here is a standard model that classifies tokens directly.

* **Model:** bert-base-cased
* **Dataset:** Standard **$K$**-shot Set A (no OADA expansion).
* **Batch Size:** **32** (on RTX 3090). Since there is no decoder, we double the batch size compared to BART.
* **Learning Rate:**  **$5 \times 10^{-5}$** . BERT usually handles slightly higher learning rates for NER than generative models.
* **Stopping Paradigm:**

  * **Minimum steps:** 50 steps.
  * **Check Interval:** Every epoch (since epochs are so short).
  * **Patience:** 5 checks.
* **Sequence length:** Truncated to 256 tokens. Matching PET research parameters.
* **Sub-Token Alignment**: soft labels are aligned with first sub-token, inter-tokens are ignored during Loss calculation.

### BERT Student (Distillation)

The Student is trained to mimic the **Ensemble's probability distribution** (Soft Labels) rather than just the hard labels.

* **Model:** bert-base-cased (or distilbert-base-cased for a lighter version).
* **Dataset:** The **full training set** (not just K-shot) with BART ensemble soft labels.
* **Distillation Loss:** **Cross-Entropy** (to match soft labels).
* **Batch Size:**  **32** .
* **Temperature (**$\tau$**):**  **2.0** . This is the standard "softening" factor used to reveal the teacher's dark knowledge.
* **Stopping Paradigm:**
  - **Minimum steps:** 1500.
  - **Check Interval:** 500 steps.
  - **Patience:** 5 checks.
  - **Sequence length:** Truncated to 256 tokens. Matching PET research parameters.
* **Sub-Token Alignment**: soft labels are aligned with first sub-token, inter-tokens are ignored during Loss calculation.

---

## Data preprocessing

1. **Data Parsing & Splitting**
2. **Pre-Tokenization and Truncation**
   Each sentence is truncated to 256 tokens (including pattern wrapping). *(PET heuristics)*
3. **Pattern Wrapping + OADA permutation & Target Generation**

## Data processing at runtime

1. **Tokenization & Collation**
   Batch is padded to equal length

## Data Splitting

### Set A: Train

The training split, is unique per run.
3 splits are created per k-value to compare different runs with the split seed varying.

* **Purpose:** **Model Fitting.** These are the primary gold labels used for gradient descent and backpropagation to update the model weights.
* **Size:** $K \times \text{classes}$ (e.g., 25, 50, 100, or 500 examples total).
* **Sampling Method:** Stratified **$K$-Shot Sampling.**
  Sampled once per seed from the official training split.

The training set is augmented with OADA / PET into baked data set unique to each ensemble.

### Set B: Mini-Val

* **Purpose:** **Trigger Early Stopping.** Monitors the training loop to detect the "peak" performance (usually via F1 score) and stops the run after 5 consecutive declining evaluations to prevent overfitting. Must be small to make scoring fast, as this is done many times during training.
* **Size:** ~100 sentences.
* **Sampling Method:** **Stratified Partition.** Sampled from the official development split with a fixed seed. Must have zero overlap with Set C or Set A.

### Set C: Tuning-Dev

* **Purpose:** **Model Selection / Internal Metrics.** Used *after* the model is fitted to compare different seeds, $K$-values, and ensemble configurations. It acts as a "proxy test set" while hyperparameters are still being dialed in.
* **Size:** ~500 sentences.
* **Sampling Method:** **Stratified Partition.** Sampled from the official development split with a fixed seed. Must have zero overlap with Set B or Set A.

### Set D: Test

* **Purpose:** **Final Report.** The "Gold Standard" evaluation for official model scoring. This represents the final generalization power of the distilled model.
* **Size:** Full (Exclusive) official Test Set (e.g., all 3,453 sentences in CoNLL-2003).
* **Sampling Method:** **Untouched.** Remained strictly unseen and locked until all training parameters and models are final.

---

## Thoughts

- **Early stopping**
  Assigning validation data to detect early stopping, is valuable in academic research.
  In applied few-shot situations you would have to rely on heuristics.
