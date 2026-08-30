# Scientific Evidence and Claims Boundaries

## Purpose

KinaBot is a dignity-first system for personal, longitudinal speech reflection.
Its design is informed by research on speech and language across aging,
repeated observation, multilingual assessment, reminiscence, and social
connection. This document records what that literature can support and, just
as importantly, what it cannot establish about KinaBot.

KinaBot has not been clinically validated as a medical device, biomarker,
screening test, diagnostic tool, treatment, or predictor of cognitive decline.
A paper about speech, aging, or reminiscence does not validate KinaBot's
implementation, scores, or outcomes.

## Product vision and testable hypothesis

KinaBot's long-term vision is to make gradual changes in a person's everyday
expression easier for that person to notice. Cognitive and communication
changes may be subtle in daily life and difficult to see without a consistent
record. KinaBot therefore aims to collect repeated, consented reflections and
visualize descriptive speech and language patterns so that a user can notice a
substantial or persistent change and decide whether to consider its context,
discuss it with family, or seek qualified professional advice.

A useful analogy is a non-diagnostic personal measurement tool: an unusual
change can prompt a question without identifying its cause. The analogy has an
important limit. Body weight is a directly measured physical quantity with
established units, while KinaBot's feature indexes are constructed,
version-specific descriptions of a speech sample. Their sensitivity,
reliability, expected day-to-day variation, clinically meaningful change
thresholds, and relationship to cognition have not yet been established.

Accordingly, the present product may help a user **notice and visualize a
change in measured speech or language features**. It must not tell the user
that the change is cognitive decline, assign a medical explanation, or imply
that any particular percentage change has clinical meaning. Developing and
testing that relationship is a future research objective, not a current
product claim.

## Evidence-to-claim map

| Evidence area | What the literature supports | Appropriate KinaBot statement | Boundary that must remain explicit |
| --- | --- | --- | --- |
| Speech and language over time | Longitudinal studies report associations between selected acoustic or linguistic features and later-life cognitive change. Speech is also relatively accessible and repeatable to collect. | Speech contains observable features that may vary over time; repeated samples can support personal reflection on those patterns. | An association is not a diagnosis or proof of causation. A KinaBot score change must not be called cognitive improvement, deterioration, or disease risk. |
| Longitudinal observation | Reviews identify repeated speech collection as promising for research while also reporting limited standardization, comparability, and clinical implementation. | KinaBot emphasizes a person's history across repeated reflections rather than drawing a conclusion from one sample. | The scientific literature does not establish that KinaBot's particular trend display is clinically meaningful. |
| Multilingual speech | Language of assessment and bilingual experience can affect language-related performance. Cross-linguistic studies indicate that some timing features transfer better than lexical-semantic features. | One account may preserve a continuous personal history across languages, while every session retains its language and language-appropriate analytical context. | Scores from different languages are not automatically equivalent. They must not be treated as interchangeable clinical measurements or direct proof of change. |
| Reflection and reminiscence | Meta-analyses of structured life-review and reminiscence interventions report potential psychosocial benefits for older adults, with variation in methods and evidence quality. | KinaBot is inspired by evidence that reflection, reminiscence, and storytelling can contribute to meaningful engagement and well-being. | KinaBot is not the intervention tested in those trials. No therapeutic benefit, treatment effect, or improvement in health can be claimed without direct studies of KinaBot. |
| Family and social connection | WHO identifies quality social connection as important to well-being and healthy aging, and social isolation and loneliness as public-health concerns. | KinaBot is designed to encourage voluntary reflection and meaningful conversations with trusted family members or care professionals. | This design intention does not prove that using KinaBot improves social connection, health, quality of life, or longevity. |

## Evidence discussed in product design

### 1. Longitudinal speech and aging

The MIDUS longitudinal work examined voice measures in relation to cognitive
change over approximately ten years. It reported associations for selected
voice features, while also finding results whose direction differed from the
authors' hypotheses. This supports cautious research interest in longitudinal
voice observation, not deterministic interpretation of a recording.

- Lin et al., *Voice Biomarkers as Indicators of Cognitive Changes in Middle
  and Later Adulthood* (2022), PMCID: PMC9487188, PMID: 35964541.
  [Full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC9487188/)
- Follow-up analysis, *Voice biomarkers in middle and later adulthood as
  predictors of cognitive changes* (2024), PMCID: PMC11527629.
  [Full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC11527629/)

A systematic review of 51 studies found promising research results for
automatic speech and language analysis in Alzheimer's disease, but emphasized
poor standardization, limited comparability, and very limited implementation in
clinical practice. This is a reason for conservative product claims.

- de la Fuente Garcia et al., *Artificial Intelligence, Speech, and Language
  Processing Approaches to Monitoring Alzheimer's Disease: A Systematic
  Review* (2020/2021), PMCID: PMC7836050.
  [Full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC7836050/)

### 2. Multilingual and cross-linguistic interpretation

A systematic review of bilingual older adults concluded that language of test
administration, cultural and linguistic background, immigration, and
acculturation need to be considered in neuropsychological assessment. Findings
across studies were not fully consistent.

- Franzen et al., *Does bilingualism influence neuropsychological test
  performance in older adults? A systematic review* (2021), PMID: 32677470,
  DOI: 10.1080/23279095.2020.1788032.
  [PubMed](https://pubmed.ncbi.nlm.nih.gov/32677470/)

A cross-linguistic English-Spanish study reported stronger transfer for speech
timing features than for lexical-semantic features. This supports retaining
language context and language-specific interpretation even when all sessions
belong to one user's continuous account history.

- Perez-Toro et al., *Automated Speech Markers of Alzheimer Dementia: Test of
  Cross-Linguistic Generalizability* (2025), PMCID: PMC12572752.
  [Full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12572752/)

The responsible interpretation for KinaBot is therefore:

> The language may change, but the person's reflection history remains
> continuous. Each language must still retain its own analytical context.

This is a product-design principle, not a clinical conclusion.

### 3. Reflection, reminiscence, and communication

A systematic review and meta-analysis of 32 studies involving 2,353 older
adults reported that structured life-review and reminiscence interventions may
improve quality of life and life satisfaction. These studies evaluated defined
interventions, not KinaBot.

- Yen and Lin, *Effectiveness on Quality of Life and Life Satisfaction for
  Older Adults: A Systematic Review and Meta-Analysis of Life Review and
  Reminiscence Therapy across Settings* (2023), PMID: 37887480.
  [PubMed](https://pubmed.ncbi.nlm.nih.gov/37887480/)

Evidence for dyadic life review involving an older person and another person is
promising but heterogeneous, and reviewers call for clearer protocols and more
consistent outcome measures.

- Ingersoll-Dayton et al., *A systematic review of dyadic approaches to
  reminiscence and life review among older adults* (2019), PMID: 30596457,
  DOI: 10.1080/13607863.2018.1555696.
  [PubMed](https://pubmed.ncbi.nlm.nih.gov/30596457/)

WHO describes social connection as important to mental and physical health and
well-being, and treats social isolation and loneliness as significant public
health concerns. This supports KinaBot's family-connection motivation; it does
not prove a health effect from the product.

- World Health Organization, *Reducing social isolation and loneliness among
  older people*.
  [WHO overview](https://www.who.int/health-topics/ageing/reducing-social-isolation-and-loneliness-among-older-people)

## Approved claim patterns

These formulations describe the present product without converting research
context into an unvalidated medical claim:

- "KinaBot helps users record personal reflections and observe descriptive
  speech and language patterns across their own history."
- "KinaBot is designed to support personal reflection and voluntary family or
  professional conversations."
- "Research on speech, aging, reminiscence, and social connection informs the
  product direction. KinaBot itself is not clinically validated."
- "Multilingual sessions remain part of one account history, while each
  session retains its language and language-specific analytical context."
- "Scores describe a sample; they are not percentages, population rankings,
  health ratings, diagnoses, or evidence of improvement or decline."
- "KinaBot aims to make otherwise subtle changes in a person's own speech and
  expression patterns easier to notice and discuss. It does not determine the
  cause or clinical meaning of a change."

## Claims that are not permitted without direct validation

Do not state or imply that KinaBot:

- detects, diagnoses, screens for, predicts, prevents, monitors, or treats
  dementia, cognitive impairment, depression, or another condition;
- produces a validated voice biomarker, cognitive score, biological age,
  disease probability, or medical risk estimate;
- proves that a user has improved or declined;
- improves cognition, health, well-being, social connection, quality of life,
  or longevity;
- is equivalent to clinical assessment or can replace qualified care;
- makes directly comparable measurements across languages without validation;
  or
- has demonstrated the outcomes reported in studies of other interventions.

## Research needed for stronger future claims

Any stronger health-related claim would require a prospectively defined study
of KinaBot itself, appropriate ethics and regulatory review, representative
multilingual participants, validated comparator measures, prespecified
endpoints, calibration and reliability testing, bias and subgroup analysis,
privacy safeguards, independent replication, and transparent reporting of both
positive and negative results.

Until then, KinaBot should remain clearly described as a non-diagnostic,
evidence-informed personal reflection tool.
