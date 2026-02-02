# Signal Specification

**Project:** Neural Schema Induction from Raw Text
**Purpose:** Define the *only* supervision injected into the system

---

## 0. Scope & Intent

This signal specification defines **how training pairs are generated** for contrastive learning.

### Non-negotiable principles

* No human labels are used
* No schema names are provided
* No field identities are assumed
* Signals are **weak, conservative, and noisy by design**
* Precision is favored over recall

Pairs exist **only to shape geometry**, not to define truth.

---

## 1. Definitions

### Token

A token is a subword unit produced by the tokenizer of the embedding model.

### Role Similarity

Two tokens are role-similar if they **consistently perform the same functional job** in sentences across contexts.

---

## 2. Positive Pair Specification

Positive pairs are token pairs that should be **pulled closer** in the learned embedding space.

### 2.1 Category A — Identity Anchors (Highest Priority)

#### Definition

Two occurrences of the **same token** across different sentences.

#### Example

```
("450" in S1, "450" in S4)
("lunch" in S1, "lunch" in S5)
```

#### Rationale

* Role consistency is maximized
* Prevents embedding drift
* Stabilizes training

#### Rules

* Always allowed
* Always included
* No filtering required

---

### 2.2 Category B — Context-Similar Numeric Tokens

#### Definition

Two numeric tokens appearing in **highly similar local contexts**.

#### Example

```
"I paid 450 for lunch"
"I paid 1200 for rent"
```

→ Positive: ("450", "1200")

#### Context similarity criteria

* Same preceding verb OR
* Same surrounding bigram/trigram
* Similar sentence position bucket

#### Rationale

* Numeric tokens are structurally stable
* Semantic similarity is irrelevant
* Role consistency is high

#### Rules

* Only digits (not number words)
* Context similarity threshold required
* Prefer mid-sentence occurrences

---

### 2.3 Category C — Frame-Equivalent Entity Tokens

#### Definition

Tokens appearing in **identical syntactic frames**.

#### Example

```
"with Rahul"
"with Amit"
```

→ Positive: ("Rahul", "Amit")

#### Frame criteria

* Same governing preposition or verb
* Same relative position
* Same local POS pattern

#### Rationale

* Names vary, roles persist
* Moderate but useful signal

#### Rules

* Must pass frame similarity
* Exclude ambiguous tokens
* Lower weight than Category B

---

### 2.4 Category D — High-Confidence Distributional Matches (Optional, Later)

#### Definition

Tokens with:

* similar frequency
* similar position distribution
* similar co-occurrence patterns

#### Rationale

* Improves recall
* Adds mild noise

#### Rules

* Disabled in early experiments
* Enabled only after geometry stabilizes

---

## 3. Negative Pair Specification

Negative pairs are token pairs that should be **kept apart**.

### 3.1 Category A — Cross-Type Role Separation (Highest Priority)

#### Definition

Tokens that almost never share a role.

#### Example

```
("450", "Rahul")
("1200", "yesterday")
("Amit", "rent")
```

#### Rationale

* Clear role incompatibility
* Very high confidence

#### Rules

* Prefer numeric ↔ non-numeric
* Prefer entity ↔ function
* Always allowed

---

### 3.2 Category B — Function vs Content Separation

#### Definition

Function words vs meaningful content tokens.

#### Example

```
("for", "450")
("with", "Amit")
("paid", "1200")
```

#### Rationale

* Prevents glue-word clustering
* Preserves structural clarity

#### Rules

* Function list must be fixed
* Content token must be non-function

---

### 3.3 Category C — Positionally Incompatible Tokens

#### Definition

Tokens with strongly disjoint sentence positions.

#### Example

```
(first token, trailing number)
(verb, terminal date)
```

#### Rationale

* Structural mismatch
* Weak but safe signal

#### Rules

* Position buckets must not overlap
* Never used alone; always combined

---

## 4. Forbidden Positive Pairs (Critical)

These pairs must **never** be used as positives.

### 4.1 Semantic Similarity Only

#### Forbidden

```
("lunch", "dinner")
("rent", "mortgage")
```

#### Reason

* Semantic similarity ≠ role similarity
* Corrupts geometry

---

### 4.2 Capitalization-Only Matches

#### Forbidden

```
("May", "Rahul")
("March", "Amazon")
```

#### Reason

* Capitalization is ambiguous
* High role variability

---

### 4.3 Rare Token Pairing

#### Forbidden

```
("Adidas", "Nike")
```

#### Reason

* Insufficient context
* High variance

---

## 5. Forbidden Negative Pairs (Critical)

### 5.1 Role-Equivalent Tokens

#### Forbidden

```
("lunch", "rent")
("dinner", "shoes")
```

#### Reason

* Both can occupy object-of-payment role
* Forces false separation

---

### 5.2 Ambiguous Tokens

#### Forbidden

```
("May", "Rahul")
("Apple", "Amazon")
```

#### Reason

* Role ambiguity too high

---

## 6. Pair Weighting (Important)

Not all pairs are equal.

### Recommended weights

| Pair Type                 | Weight |
| ------------------------- | ------ |
| Identity Anchors          | 1.0    |
| Numeric Context Positives | 0.9    |
| Frame-Equivalent Entities | 0.7    |
| Distributional Matches    | 0.5    |
| Cross-Type Negatives      | 1.0    |
| Function Negatives        | 0.8    |
| Positional Negatives      | 0.6    |

---

## 7. Safety Rules (Non-Negotiable)

* If uncertain → discard the pair
* No token appears in both positive and negative with same anchor
* Positive precision > negative precision
* Pair quantity is secondary to correctness

---

## 8. Success Criteria for Signal

This signal is considered **valid** if:

* Training converges without collapse
* Numeric tokens cluster coherently
* Entities cluster separately
* Function words do not form schema clusters
* Geometry is stable across random seeds

If these fail → revise signal before touching the model.

---

## 9. Explicit Non-Goals

This signal does **not** aim to:

* fully classify tokens
* resolve ambiguity
* extract production schemas
* eliminate noise

It exists **only** to inject role bias.

---

## Final Statement (Do Not Skip)

> **This signal does not encode truth.
> It encodes pressure.
> Structure must emerge, not be imposed.**

---
