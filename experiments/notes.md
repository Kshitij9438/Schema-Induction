# Experiment Notes — Schema Induction Project

## Context

This document records observations, failures, and design conclusions from iterative experimentation on the schema-free role induction and extraction system.

The goal is not to justify outcomes, but to **understand the boundaries of the architecture** and make principled decisions.

---

## Path A Summary — Baseline Capability

### What Worked

- **Amount role**
  - Extremely stable across all tests
  - Survives word reordering, telegraphed inputs, and surface noise
  - Behaves as an atomic, competitive role

- **Item role**
  - Moderately stable
  - Extracted reliably when sufficient context exists
  - Drops conservatively under extreme surface stress
  - Never hijacks amount

These behaviors confirm that **token-level role induction is sufficient for atomic entity/value roles**.

---

## Path B — Surface-Form Robustness Experiments

### Experiment Setup

- Sentence templates were heavily diversified:
  - Inverted word order
  - Telegraphed phrases (e.g. `"450 lunch"`)
  - Verb-dropping
  - Time-leading and person-leading forms
- No retraining logic or model architecture was changed
- Only data surface form was perturbed

### Observations

- Learned role geometry did **not collapse**
- Numeric tokens remained tightly clustered
- Item tokens remained moderately clustered
- No catastrophic mixing of roles occurred

This indicates that the model did **not memorize templates**, but learned structural role signals.

---

## Time Role — Critical Finding

### Empirical Behavior

- Temporal tokens (`today`, `yesterday`, `morning`, `night`) appear in role clusters
- However, time extraction at token level is:
  - Inconsistent
  - Low confidence
  - Easily overshadowed by amount/item roles

Attempts to improve time extraction via:
- Threshold tuning
- Policy relaxation
- Non-competitive role handling

did **not** yield stable improvements.

### Key Insight

> **Time is not an atomic token-level role.  
> It is a contextual, span-level modifier.**

This explains why:
- `amount` works extremely well (atomic, numeric, rigid)
- `item` works reasonably well (noun-like, semi-atomic)
- `time` does not work reliably (relational, compositional, diffuse)

The system is behaving honestly by refusing to hallucinate time when context is insufficient.

---

## Architectural Boundary Identified

The current system assumes:

```

role = f(token_embedding)

````

This assumption is **valid for entities and values**, but **invalid for modifiers** like time.

Proper time extraction would require at least one of:
- Span-level representations (`"last night"`)
- Event anchoring (time modifies actions, not tokens)
- Relational or syntactic context

These are **out of scope** for the current architecture by design.

---

## Design Decision — Time as Optional Metadata

Given the above, the following decision was made:

- Treat **time as optional metadata**, not a core schema field
- Extract time only when trivially obvious
- Prefer omission over incorrect assertion

This preserves:
- Precision
- Interpretability
- Architectural honesty

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

Time may be absent, and that is considered **correct behavior**.

---

## Final Conclusion

* The system successfully demonstrates **unsupervised schema induction** for atomic roles
* Robustness testing revealed a **natural representational boundary**
* Failures are explainable and principled, not random or buggy
* The architecture’s limitations are clearly understood and documented

This is considered a **successful and complete outcome** for the current scope.

Future extensions (v2) may explore span-aware or relation-aware modeling, but are intentionally excluded here.

