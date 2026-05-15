# Few-NERD Data Setup

Few-NERD =  **Few-shot Named Entity Recognition Dataset** .

Specifically designed for:

* few-shot learning
* low-resource NER
* generalization to unseen entity types
* **8 coarse entity types** - Tom [Person], Paris [Location]

* **66 fine-grained entity types** - Tom [Person-actor], Paris [Location-GPE]

**Data scource**: `DFKI-SLT/few-nerd` (Hugging Face version of the Few-NERD dataset)

Data size:

Validation: 18,824 examples

Test: 37.648 examples

Train: 131,767 

### Current setup

`DFKI-SLT/few-nerd`

Fine-grained 

Supervised 

Converted to BIO-tagging

mini-val: 200 examples (~1% of validation size)

### 3 configs

**Supervised**

Train/validation/test all contain overlapping entity types.

This is the closest to CoNLL-style training.

Example:

train contains `person-actor`

test also contains `person-actor`

**Intra**

Train/test share the SAME coarse type, but different fine types.

Example:

Train:

* person-actor
* person-athlete

Test:

* person-politician

So:

* the model has seen `person`
* but not that specific subtype

This tests:

* fine-grained transfer within a semantic family

**Inter**

Train/test contain DIFFERENT coarse types.

Example:

Train:

* person
* location

Test:

* product
* event

This tests:

* true cross-domain few-shot generalization

This is much more meta-learning/few-shot research territory.

### IO tagging

Few-NERD uses IO taggin, which is different from ConLL2003's BIO tagging



| IO                  |                     | BIO                 |
| ------------------- | ------------------- | ------------------- |
| ***Token*** | ***Label*** | ***Label*** |
| Barack              | PER                 | B-PER               |
| Obama               | PER                 | I-PER               |
| visited             | O                   | O                   |
| New                 | LOC                 | B-LOC               |
| York                | LOC                 | I-LOC               |

**Why does this matter?**

Because BIO is easier for:

* boundary detection
* span reconstruction
* exact entity extraction

IO is:

* simpler
* more compact
* but slightly less informative


**Converting IO → BIO does NOT lose information.**


## Warning!

Few-NERD **fine** labels are MUCH more imbalanced than CoNLL which affects:

* greedy k-shot sampler
* class coverage
* OADA permutations
* evaluation variance

This leads to:

* K=5 is extremely unstable
* some labels missing entirely

**Suggestions for later:**

* minimum class coverage checks
* per-class statistics
* filtering ultra-rare classes
