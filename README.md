# Schema-Induction: Unsupervised Role Learning from Raw Text

This project explores **schema-free role induction** from unstructured text using **no labels, no predefined schemas, and no task-specific supervision**.

Instead of predicting fields directly, the system learns a **latent role geometry** and uses it to extract structured information in a principled, interpretable way.

---

## Problem Statement

Most information extraction systems assume the schema upfront:

- Named Entity Recognition (NER)
- Slot filling
- Rule-based pipelines
- Prompt-engineered LLM extraction

These approaches require:
- predefined field names
- labeled data
- brittle rules
- or hidden supervision

This project asks a harder question:

> **Can structure emerge from raw text without knowing the schema in advance?**

---

## Why This Is Hard

Unsupervised extraction fails for three reasons:

1. **Semantic similarity ≠ role similarity**  
   (“lunch” and “dinner” are semantically similar, but their *role* is what matters)

2. **Tokens are noisy and ambiguous**  
   Function words, verbs, and modifiers dominate text

3. **Most roles are implicit, not labeled**  
   Especially values, objects, and modifiers

This system explicitly separates **learning structure** from **making extraction decisions**.

---

## System Overview

High-level flow:

```

Raw text
↓
Frozen language model (semantic baseline)
↓
Trainable role projection head
↓
Latent role embedding space
↓
Unsupervised clustering
↓
Policy-based extraction

````

Key idea:
> **Learn geometry first. Decide meaning later.**

---

## Learning Signal (No Labels)

The system uses **contrastive learning with weak, structural signals**, not labels.

Examples of signals:
- Same token across contexts → positive
- Numeric tokens in similar frames → positive
- Numeric vs non-numeric → negative
- Function words vs content → negative

These signals:
- do **not** encode schema
- do **not** encode truth
- only apply **geometric pressure**

See `docs/SIGNAL_SPEC.md` for the exact specification.

---

## Role Induction

A small projection head is trained on top of a frozen encoder to reshape the embedding space.

Observed emergent roles:
- **Amount** (numeric, atomic, highly stable)
- **Item** (object of transaction, moderately stable)
- **Person** (participant, subword-fragmented)
- **Time** (contextual, diffuse)

Clustering reveals that some roles are **naturally atomic**, while others are **relational**.

---

## Robustness Experiments (Path B)

To test whether roles were learned or memorized, surface form was aggressively perturbed:

- Inverted word order
- Telegraphed phrases (`"450 lunch"`)
- Verb dropping
- Time-leading phrases (`"yesterday 500 coffee"`)

Results:
- Role geometry remained stable
- Numeric roles never collapsed
- Item roles degraded gracefully
- No catastrophic mixing occurred

This indicates the model learned **structural roles**, not templates.

---

## Policy Layer (Inference-Time Decisions)

Learning does **not** decide what gets emitted.

A separate **policy layer** handles:
- type sanity (e.g. digits for amounts)
- subword hygiene
- role sparsity (one amount, one item)
- conservative thresholds

This keeps learning honest and extraction interpretable.

---

## Design Decision: Time as Optional Metadata

A key experimental finding:

> **Time is not an atomic token-level role.  
> It is a contextual, span-level modifier.**

Attempts to force time into token-level extraction:
- reduced precision
- caused false positives
- conflicted with learned geometry

Final decision:
- Treat **time as optional metadata**
- Extract only when trivially obvious
- Prefer omission over incorrect assertion

Example output:

```json
{
  "amount": "450",
  "item": "lunch",
  "meta": {
    "time": "yesterday"
  }
}
````

This reflects an architectural boundary, not a bug.

---

## What This System Is (and Is Not)

### This system **is**:

* Unsupervised
* Schema-free
* Interpretable
* Robust to surface variation
* Explicit about its limits

### This system **is not**:

* A production extractor
* A replacement for NER
* Span- or syntax-aware
* Optimized for recall

---

## Key Takeaway

> **Token-level role induction works well for atomic roles (values, objects),
> but contextual modifiers (time) require span- or relation-level modeling.**

Identifying this boundary is the main contribution of this project.

---

## Future Work (v2 Ideas)

* Span-aware role representations
* Event-centric modeling (time anchored to actions)
* Multi-token merging (`"last night"`)
* Domain transfer beyond expenses

These are intentionally out of scope for the current system.

---

## Repository Structure

```
data/
  synthetic/        # Controlled data generation
  pairs/            # Contrastive signal construction

models/
  encoder.py        # Frozen semantic encoder
  role_head.py      # Trainable role projection

training/
  train.py          # Contrastive training loop

analysis/
  cluster.py        # Unsupervised role clustering
  extract.py        # Policy-based extraction

docs/
  SIGNAL_SPEC.md    # Exact learning signal definition

experiments/
  notes.md          # Experimental observations & conclusions
```

---

## Final Note

This project is deliberately **honest**.

Instead of hiding failure cases, it:

* exposes them
* explains them
* uses them to reason about architectural limits

That honesty is the point.


