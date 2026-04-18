# Poznámky z přednášky k zadání projektu

Syntéza z přepisu přednášky, kde Lukáš Burget prezentoval zadání. Zachyceno
je všechno, co se nedá vyčíst z `assignment.txt` — tj. věci řečené ústně,
varování, preference při hodnocení, konkrétní pasti. Je to zdroj faktů, ze
kterého budeme čerpat při psaní `dokumentace.pdf` a při obhajobě.

---

## 1. Co jsou data

- Každý účastník byl **natočen na video**, během kterého četl věty nebo
  odpovídal na otázky. Přicházel se nahrávat **opakovaně, cca jednou za
  měsíc** — tím vznikají různá *sessions*.
- **PNG = jeden frame z toho videa.** **WAV = zvuková stopa z toho videa.**
  Obraz a zvuk jsou proto dokonale spárované na úrovni události (1:1).
- Celkem cca **30–32 lidí** napříč datasetem.
- **Target = `m431` = Ondra Glembeck** (bývalý kolega na FIT). Mezi non-target
  figuruje i **Silvie Sadoská**.
- **Obrázky jsou 80 × 80 px.** V demu je konvertuje na šedotón součtem RGB.

### Klíčová terminologie

| Pole v názvu souboru | Význam                                                    |
| -------------------- | --------------------------------------------------------- |
| `identity` (`m431`)  | Osoba                                                     |
| `session` (`01`)     | Jeden den nahrávání — sdílí osvětlení, pozadí, mikrofon   |
| `prompt` (`p03`)     | Konkrétní věta / otázka čtená během session               |
| `i<inst>_<take>`     | Pokus v rámci session; skoro vždy `i0_0`                  |

**V rámci jedné session** má člověk stejné oblečení, pozadí, osvětlení, vousy,
brýle, náladu, mikrofon. **Mezi sessions** se všechno tohle může lišit.

## 2. Co bude v ostré evaluaci

- Data přijdou **v neděli 3. května ráno**.
- **Řádově ~1000 souborů.** Student je jen proženne svým už hotovým skriptem.
- **Obsahem budou:**
  1. originální vzorky (typově stejné jako trénovací),
  2. **nové sessions** těch stejných lidí (Burget explicitně: *"některé jsem
     si nechal do zásoby"*),
  3. vzorky **schválně poškozené / augmentované** — šum, změny kvality,
     artefakty.
- Burget o tom řekl: *"je to pro vás loterie"* — kdo se trefí do toho, jak on
  vzorky zprasil, dostane lepší čísla. Ale také: *"ti, co se víc snažili a
  udělali něco sofistikovanějšího, tomu to nakonec funguje líp"*.
- **Vypnuté pretrained modely mají přesně tenhle účel** — aby studenti
  nemohli natáhnout embedding z internetu a mít 100 %. Burget záměrně
  simuluje low-data prostředí.

## 3. Formát výsledků — pasti, za které strhává body

ASCII soubor, jeden řádek na vzorek, tři pole oddělená mezerou:

```
<stem> <score> <hard_decision>
```

### `stem` (první pole)
- **Bez přípony.** Každý rok ~10 % studentů tam přípony má → Burget pak
  ručně maže → **strhává body**.

### `score` (druhé pole)
- Reálné číslo. **Vyšší = detektor si je jistější, že je to target.**
- Ideálně **log-likelihood ratio** nebo **logit(P(target|x))** — tedy hodnota
  **před finální sigmoidou** u logistické regrese nebo neuronové sítě.
- **Nikdy jen 0/1** — nedá se oskórovat jako ROC/EER metrika.
- **Vyhni se syrové pravděpodobnosti 0..1**, obzvlášť když je systém tak
  nastavený, že produkuje saturované hodnoty (skoro-0 nebo skoro-1).
- Přidaná hodnota: taková skóre se **dobře fúzují** — můžeš je prostě
  zprůměrovat napříč systémy.

### `hard_decision` (třetí pole)
- `1` = target, `0` = jinak.
- **Předpokládej apriorní pravděpodobnost 0.5**, i když reálně většina
  testovacích vzorků target nebude. Burget při hodnocení aplikuje tento
  prior sám (ekvivalent *balanced accuracy* přes třídy).

### Názvosloví souborů
- **ZIP** přesně `login1_login2.zip` (nebo `login1.zip` pro sólistu).
- **Dokumentace** přesně `dokumentace.pdf` — ne `documentation.pdf`, ne
  `dokumentace.pdf.pdf`. Jméno je load-bearing pro jeho skripty.
- **Výsledkové soubory** pojmenuj sebepopisně: `audio_gmm.txt`,
  `image_pca_logreg.txt`, `fusion_logreg.txt`.
- **Max 6 výsledkových souborů** bude skórováno. Můžeš poslat víc, ale v
  dokumentaci označ 6 hlavních.

## 4. Self-test — povinný krok před odevzdáním

> "Lidi mě posílají soubor, kde jména souboru jsou správně, ale skóre jsou
> k nim přeházené, protože měli blbý hash v Pythonu."

- Vytvoř si **vlastní mini-eval sadu** s **known ground truth**.
- Proženni ji celým pipelinem.
- **Ověř, že pořadí skóre odpovídá pořadí stemů.**
- Studenti tohle řeší v poslední den a nestíhají.

## 5. Validační strategie — konkrétní pokyny

- Burget doporučuje **3 nebo 4 foldy**, ne 10. Target má jen 3 sessions →
  víc foldů nedává smysl.
- **Session-aware** na cílové osobě: *"v rámci jedné session vypadá obličej
  často velmi podobně, velmi lehce se model naučí říct ‚to jsem už viděl'"*.
- **Speaker-aware** na non-target: některé lidi do trainu, jiné do valu.
- **Prompt (třetí pole) asi není důležité.** Řídit se můžeš, ale nemusíš.

### Co s CV výsledkem (4 legitimní cesty)

Po doběhnutí CV můžeš:

1. **Vzít jeden ze K modelů** (ten z nejlepšího foldu).
2. **Zprůměrovat parametry** napříč foldy (kde to dává smysl — lineární
   modely, GMM průměry).
3. **Nechat všech K modelů a průměrovat výstupní skóre** (ensemble).
4. **Zafixovat hyperparametry a přetrénovat na všech datech dohromady.**

U nás nejpraktičtější bude **3 (ensemble)** nebo **4 (retrain na plných
datech)**. Rozhodni podle toho, co CV skóre ukáží.

## 6. Dokumentace — co ji zachrání

- Musí vysvětlit **PROČ**, ne jenom **JAK**. Opravující dostane tuto
  instrukci explicitně.
- Povinně musí být vidět:
  - **vývojové úvahy** — co jsi zkoušel, co zafungovalo, co ne,
  - **validační strategie** a její zdůvodnění,
  - **řešení generalizace a overfittingu** (augmentace, regularizace),
  - **jak reprodukovat výsledky**.
- Burgetova explicitní preference: **"Když uvidíme, že jste byli odvážní a
  zkusili něco, co nakonec nevyšlo, dostanete víc bodů, než když jste to
  odbili a výsledek je průměrný."**

### Ablace jsou tvůj kamarád

Pro každou techniku, kterou zdůvodňuješ (augmentace, feature, model), ukaž v
reportu **tabulku** "bez / s → EER klesl z X na Y". To je jediný pádný důkaz,
že ta technika pomáhá.

## 7. Ústní obhajoba

- **Termín:** první týden zkouškového. Burget zvažuje přifařit k první
  zkoušce ze SUR (odpolední sloty tentýž den).
- Opravující dostane **tvoji dokumentaci a kód dopředu** a má se na ně
  podívat.
- Umí se zeptat na detaily: proč jsi volil X, jak funguje Y.
- Burget explicitně: **"Předpokládám, že možná použijete LLM. Ale musíte
  tomu rozumět a obhájit každé rozhodnutí."**

## 8. Styl kódu — silné preference

- **Jednoduše.** Burget řekl doslova: *"Čím jednodušší, tím lepší."*
- **Ne** notebook pouštějící externí skripty, které pouštějí další skripty.
- **Ne** OOP kvůli OOP. **Ne** cluster-ready submitter.
- **Ano** přímočaré skripty, které jdou přečíst a pochopit.
- **Velké neuronky = špatný nápad.** Burget varoval: *"obrovská neuronová
  síť se ti na tak malém množství dat přetrénuje"*. Lehké modely vyhrají.

## 9. Augmentovaná data

- **Neposílej je v ZIPu.** Je v tom explicitní pokyn.
- Místo toho **skript**, který z originálních dat vyrobí augmentované verze.
- Burget počítá s tím, že augmentace je stochastická → nebudeme umět
  zreplikovat číselně přesně, ale plus mínus ano.

## 10. Co přináší body (podle toho, co zdůraznil)

1. **Augmentace** s viditelným přínosem v ablaci.
2. **Session/speaker-aware CV** (3–4 foldy).
3. **Model nad rámec baselinu** s odůvodněním, proč nepoužíváme velký CNN.
4. **Fúze se zdůvodněním** (score-level vs. feature-level).
5. **Reprodukovatelnost + self-test** s vlastní mini-eval sadou.
6. **Skóre jako log-ratio**, ne 0/1. Fúze pak line rovnou funguje.

## 11. Timeline

| Datum              | Co se děje                                          |
| ------------------ | --------------------------------------------------- |
| neděle 3. 5. ráno  | Eval data vypuštěna                                 |
| pondělí 4. 5. 23:59 | **Deadline odevzdání v IS**                        |
| úterý 5. 5.        | Klíč zveřejněn                                      |
| středa 6. 5.       | Poslední přednáška — analýza výsledků, vyhlášení    |
| 1. týden zkoušk.   | Ústní obhajoba (nejspíš odpoledne po první zkoušce) |

**Burget řekl jasně:** *"poslední den to už nestihnete. Musíte to mít v té
době hotové a dolaďovat jen dokumentaci."*

## 12. Bonus

Za **nejlepší obrázkový systém**, **nejoriginálnější systém** atd. dává
**kvalitní láhev vína** (s medailemi z vinařských soutěží). 2–3 lahve na
poslední přednášce. Kdo dělá sám, soutěží jako jednotlivec.

---

## 13. Demo, které na přednášce ukázal (image baseline)

Přednáška skončila uprostřed image demo; audio baseline už tam není. Co tam
proběhlo:

- Načte PNG → šedotón (součet RGB kanálů) → flatten 80×80 → vektor 6400.
- Trénovací matice `X ∈ R^{152 × 6400}` (132 non-target + 20 target).
- Labely: `[1]*20 + [0]*132`.
- Spočte **PCA přes SVD** matice `X - mean(X)` (ne přes kovarianční matici).
- Rekonstrukce obličeje z prvních `N` vlastních vektorů → demonstrace, že
  **20–140 dimenzí stačí** k zachování informace o tom, kdo to je.
- Ukázková interpretace vlastních vektorů: první zachycují **osvětlení a
  pozadí**, další přidávají detaily (brýle, tvar obličeje).
- Pak měl na plánu použít těch ~20 dim jako feature pro klasifikátor.

Na přednášce existuje `ikrlib.py` s pomocnými funkcemi (`png2feat`,
`wav2mfcc`, `train_gmm`, `logistic_regression_fit`, …). **Nemusíme je
používat**, ale můžeme — soubor je v `demos/` ze staženého archivu.
