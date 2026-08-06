# Origin, Collaboration, and an Independent Product Boundary

**Period:** May 2025 to June 2026

**Status:** Retrospective case based on preserved repository history

**Source:** Founder record, Git history, and maintained-project documentation

## Background

KinaBot's maintained founder record traces its product origin and MVP launch to
an early concept called Kizuna in May 2025. This is provenance supplied by the
founder, not a Git timestamp. The repository's preserved history across all
refs begins on 2025-11-28; the current maintained branch contains a later
reconstructed history beginning on 2026-06-11. The early work explored
speech-derived cognitive insights, Streamlit delivery, privacy
notices, testing, reports, and frontend/backend separation. It includes
identifiable contributions from Irene Li and Yuan Chen as well as Aoi Minamoto's
founding, integration, maintenance, and product direction.

The early materials are preserved because provenance matters. They also contain
concepts such as cognitive age, overall scores, and risk-style presentation that
do not define the maintained product today.

## Trigger

As the work moved from exploration toward a product that people might use for
healthy-aging reflection, historical experiments and current behavior could no
longer share an ambiguous boundary. Readers needed to know who contributed,
what Aoi currently maintained, and which older claims were no longer accepted.

## Decision

On 2026-06-11, Aoi created `aoi_kinabot_app/` as a separately documented new
version and independent technical direction. Historical collaborative work and
all identifiable contributors remained visible outside that directory. The Git
record for `aoi_kinabot_app/` attributes its development and maintenance to Aoi
Minamoto; the new directory established its own plans, privacy model, scoring
design, tests, database helpers, interface, and contribution rules.

This does not convert historical collaborative work into Aoi's sole work. It
creates an auditable boundary around the new application that Aoi independently
developed, directed, and continues to maintain.

## Maintainer Reflection

> I learned that leadership is not demonstrated by removing other people's
> history. It is demonstrated by preserving provenance, defining the new
> responsibility clearly, and continuing to make verifiable decisions over
> time. A clean boundary made both collaboration and independent contribution
> easier to understand.

## Reusable Knowledge

- Preserve early work even when the product direction changes.
- Separate founder, maintainer, author, owner, and inventor; they are not the
  same legal or technical role.
- Establish a maintained boundary with code, documentation, tests, and decision
  authority—not merely a new product name.
- Attribute collaborators precisely and never use later leadership to erase
  earlier contributions.

## Evidence and Limits

Evidence includes the current repository history, the 2026-06-11 commit
[`8a92fd8`](https://github.com/usekina/kina/commit/8a92fd8), and
[`OWNERSHIP-AND-MAINTAINERSHIP.md`](../../../OWNERSHIP-AND-MAINTAINERSHIP.md).
The broader chronology is documented in
[`PROJECT-EVOLUTION.md`](../../PROJECT-EVOLUTION.md).

The May 2025 MVP launch should be supported separately by contemporaneous records
if it is used in a formal proceeding. The repository supports a record of
contribution and continuity from its preserved commit history. It does not,
by itself, determine patent inventorship, prove exclusive authorship of all
historical work, or establish field-wide impact.

## Discussion Questions

1. How can a founder preserve collaboration while establishing a new product
   boundary?
2. Which artifacts show ongoing technical leadership better than commit count?
3. When does a redesign become a distinct maintained system rather than a
   continuation of an experiment?
