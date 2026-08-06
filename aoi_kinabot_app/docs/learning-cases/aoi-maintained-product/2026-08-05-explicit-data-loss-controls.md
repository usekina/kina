# Users Must Never Have to Guess About Data Loss

**Date:** 2026-08-05

**Status:** AImoji product principle documented

**Source:** First-hand product reflection by Aoi Minamoto

## Trigger

An interface can cause real harm even when its underlying data operation is
technically correct. If a button, icon, toggle, navigation step, replacement
flow, or timeout can remove user data, the user may discover the consequence
only after recovery is impossible.

This is especially important in human-centered and health-adjacent products,
where a record may contain personal effort, reflection, family context, or
sensitive information that cannot simply be recreated.

## Founder Principle

> 任何会导致用户数据丢失的操作，都必须有明确的按钮、明确的提示，并且不能依赖用户猜测。

> Any operation that can cause the loss of user data must have an explicit
> control and an explicit warning. It must never depend on the user guessing
> the consequence.

## Product Requirements

A user-facing destructive action must provide:

- a distinct, intentionally labeled control;
- a plain-language explanation of exactly what will be lost;
- a statement about whether recovery is possible;
- confirmation proportionate to the severity of the loss; and
- a safe default that preserves data.

Destructive behavior must not be concealed behind an ambiguous icon,
navigation, toggle, timeout, or unrelated primary action.

## Important Boundary: Privacy Deletion

Not every deletion should require a user to press a delete button. KinaBot, for
example, removes temporary audio after processing to minimize sensitive-data
retention. Such automatic lifecycle deletion is appropriate when it is:

- disclosed before data collection;
- necessary for a stated privacy or retention purpose;
- applied consistently on success and failure paths; and
- verified by tests and operational records.

This distinction prevents a transparency principle from accidentally weakening
privacy-by-design behavior.

## Maintainer Reflection

> I learned that a technically available function is not an understandable
> choice. If people must guess whether an action will erase their work, the
> interface has transferred product responsibility to the user. Dignity means
> making the consequence visible before the loss occurs.

## Reusable Knowledge

- Treat destructive actions as a separate interaction class.
- Name the affected data, not merely the operation.
- Match confirmation friction to severity and reversibility.
- Test comprehension and recovery, not only whether deletion executes.
- Distinguish user-initiated destruction from disclosed retention automation.
- Record deletion events without logging the sensitive content being deleted.

## Student Exercise

Choose a product flow involving delete, reset, overwrite, replacement, account
closure, retention expiry, or import. Produce:

1. a data-loss inventory;
2. button and warning copy;
3. confirmation and recovery behavior;
4. accessibility and localization requirements;
5. tests for accidental activation and failure recovery; and
6. a separate analysis of any privacy-required automatic deletion.

## Discussion Questions

1. When does confirmation protect a user, and when does it become habituating
   friction?
2. Which losses require undo, a recovery window, or administrator support?
3. How should destructive-action warnings work for older adults, screen-reader
   users, and multilingual audiences?
4. When can privacy-preserving automatic deletion occur without an immediate
   button press?

## Evidence and Claims Boundary

The formal rule is recorded in
[`docs/maintainer-principles.md`](../../../../docs/maintainer-principles.md) and its
initial implementation record is commit
[`386bf0f`](https://github.com/usekina/kina/commit/386bf0f).

This case documents a product-governance principle. It does not establish that
every current interface already satisfies the rule. Product-specific audits and
tests are required before making that claim.
