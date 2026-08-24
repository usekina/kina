# Authorship and Rights Provenance

Last reviewed: 2026-08-24

This record separates observable Git history from legal conclusions. It is an
engineering audit, not a determination of copyright ownership or legal advice.

## Maintained Application

Git history currently attributes commits under `aoi_kinabot_app/` to Aoi
Minamoto identities. The project policy now permits only approved Aoi personal
or AImoji project identities to author and commit changes in that directory.
External reports, research, design, testing, and patch proposals must still
receive accurate non-code credit.

Commit `751ecd368ccb6f82ba13bd60ad1c1e7afd0e6418` was recorded with the author
identity `Aoi Minamoto (TEMA) <aoi.minamoto@toyota.com>`. The maintainer identifies
this as an unintended Git metadata selection, not a Toyota contribution,
sponsorship, affiliation, or endorsement. `.mailmap` maps generated contributor
views to Aoi's canonical GitHub identity while the immutable commit object is
retained for audit integrity.

This factual correction does not by itself resolve any employment or copyright
question. The maintainer should privately retain evidence that the work was
independently created and authorized, and seek counsel if commercial licensing
relies materially on it.

## Historical Repository Materials

Repository-root history includes commits attributed to Aoi Minamoto, Yuan Chen,
and IreneLi. Those historical exploratory materials must not be represented as
solely authored or solely owned by AImoji without documented rights evidence.
The public Git history remains the attribution record unless corrected by the
relevant contributor or another reliable source.

## Prospective Contributions

- `CONTRIBUTING.md` defines submission and identity rules.
- `CLA.md` applies prospectively after adoption and affirmative acceptance.
- `CODEOWNERS` identifies required maintainers but does not transfer copyright.
- CI blocks specified unrelated employer domains and unauthorized commit
  identities in `aoi_kinabot_app/`.

## Review Before Commercial Reliance

Before offering a commercial license that includes historical or externally
contributed material, review:

- the license present when each contribution was submitted;
- pull-request text, contributor agreements, and employer authorization;
- third-party dependencies, copied content, datasets, models, and assets;
- whether a component can be licensed by AImoji or must retain separate terms;
- whether independent reimplementation or contributor permission is required.

Do not alter public Git history merely to create the appearance of cleaner
ownership. Correct display metadata with `.mailmap`, preserve evidence, and
document legal resolution separately.
