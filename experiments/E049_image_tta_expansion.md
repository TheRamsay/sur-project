# E049 — Image TTA Expansion — Implementation Mismatch

- **Date:** 2026-04-22
- **Author:** TheRamsay
- **Related:** E043 (image TTA flip+rot5, 0.74% EER)

## Hypothesis

More TTA views (rot7, rot9, brightness, noise) will improve over E043's 5-view TTA.

## Result

| TTA Type | Views | EER | vs E043 |
|----------|-------|-----|---------|
| E043 (replication) | 5 | 4.38% | +3.64pp ❌ |
| rot7 | 7 | 4.59% | +3.85pp ❌ |
| rot9 | 9 | 4.59% | +3.85pp ❌ |
| bright | 7 | 4.59% | +3.85pp ❌ |
| noise | 7 | 4.59% | +3.85pp ❌ |

**ALL configurations catastrophically worse than original E043 (0.74%)!**

## Root Cause

**Implementation mismatch with E043:**

1. **Adversarial training not replicated correctly**
   - E043 used specific adversarial rotation training (±10°)
   - E049 training augmentation differs slightly
   - Fold 0 pathology returns (10.71% across all configs)

2. **E043 baseline not properly replicated**
   - Original E043: 0.74% EER
   - E049 E043 replication: 4.38% EER
   - 6× degradation indicates fundamental implementation difference

3. **TTA expansion irrelevant when base model is broken**
   - Can't improve TTA when training is wrong
   - Garbage in, garbage out

## Key Insight

**E043's 0.74% EER is fragile** — small changes to training augmentation break it completely. The adversarial training setup is critical and must be replicated exactly.

## Decision

**REJECTED.** E049 implementation doesn't match E043. TTA expansion study is invalid.

**Keep E043 as image flagship at 0.74% EER.**

Future work: Fix E049 implementation to exactly match E043, then test TTA expansion.
