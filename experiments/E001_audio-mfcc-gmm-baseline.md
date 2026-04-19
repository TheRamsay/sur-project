# E001 — Audio baseline: MFCC + GMM

- **Date:** 2026-04-19
- **Author:** Dominik Huml
- **Related experiments:** —

## Hypothesis

13 MFCC koeficientů s CMN, dva oddělené GMM modely (target a non-target),
skóre jako LLR. Očekáváme EER v rozmezí 15–30 % — baseline by neměl být
skvělý, ale měl by být lepší než náhoda (EER = 50 %).

## Setup

- **Modality:** audio
- **Data:** train + dev (combined), 222 vzorků (30 target, 192 non-target)
- **Features:** MFCC 13, CMN (per-utterance), bez delta
- **Model:** dva GMM — target (8 komponent), non-target (32 komponent)
- **Fold spec:** `iter_folds_loso`, seed=67, 3 foldy (LOSO na target sessions)
- **Seed:** 67
- **Command / notebook:** `notebooks/E001_audio_gmm.ipynb`
- **Augmentation:** none

## Result

| Fold | EER [%] | min-DCF |
| ---- | ------- | ------- |
| 0    | 20.42   | 0.3083  |
| 1    | 24.17   | 0.1833  |
| 2    | 9.17    | 0.1833  |
| mean ± std | 17.92 ± 7.81 | 0.2250 ± 0.0722 |

OOF celkové: EER = 14.48 %, min-DCF = 0.2677, threshold = −1.267

## Interpretation

Hypotéza potvrzena — systém je lepší než náhoda (EER < 50 %).
Fold 2 (session 03) výrazně lepší než foldy 0 a 1 — pravděpodobně session 03
je kvalitativně blíže trénovacím session. Velký std (7.81 %) říká, že výsledek
silně závisí na tom, která session je na valu — to je realistický odhad
variability na ostrých datech.

Threshold −1.267 (ne 0) naznačuje, že skóre nejsou dobře kalibrovaná —
GMM pro non-target je "příliš jistý". Kalibrace by pomohla.

## Next step

- E002: přidat delta + delta-delta MFCC → očekáváme zlepšení o 3–5 % EER
- E003: GMM-UBM + MAP adaptace (flagship audio systém)
- Zvážit kalibraci skóre (Platt) na OOF
