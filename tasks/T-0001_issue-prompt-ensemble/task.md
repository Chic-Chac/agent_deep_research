---
id: T-0001
type: task
title: "Prompt ensemble design for stance-neutral issue sentences"
status: done
requester: user
received: 2026-11-06T11:50+09:00
finished: 2026-11-06T12:02+09:00
deliverable: R-0001
tags: [prompt-ensemble, issue-normalization, clustering]
---

# Prompt ensemble design for stance-neutral issue sentences

## Request

Backfilled 2026-11-11; original request text not recorded. Topic (from report): which prompt-ensemble design and aggregation suit an LLM step that rewrites text blocks into stance-neutral issue questions that are then embedded and clustered.

## Scope

Not recorded (backfilled). Sub-questions: see N-0001..N-0005.

## Outcome

- Vary the route to the answer (exemplar sets, decomposition, claim-then-neutralize, context on/off); hold target granularity, output form, neutrality rule fixed.
- Avoid persona/perspective and abstraction-level variation.
- Aggregate by medoid (MBR) selection, not embedding averaging; use member dispersion as confidence.
- Open: no published study ensembles prompts for a normalization output that is then embedded and clustered; transfer is by analogy.
