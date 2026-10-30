# Orange theme rollout

Production verification address: https://d1wcbnm8yunv7y.cloudfront.net/

Merging code does not rebuild or replace the running AWS container. The current
GitHub workflows run checks, not a production deployment. Keep source merge and
live deployment verification separate.

This change uses orange-red accents with warm neutral text and surfaces. Native
Streamlit controls inherit the same theme when launched from the repository
root or application directory. Explicit orange is also set for the history line
chart; the radar uses orange shades and dashed references. Alert type labels and
icons remain distinguishable even though their palette is unified.

Container builds accept BUILD_REVISION, exposed as the OCI image revision label
and KINABOT_BUILD_REVISION environment variable. Deployment scripts pass the
checkout's full source commit. Match that revision to the intended GitHub commit.

For an existing production service, inspect the current CloudFront origin, ECS
service and task definition before changing anything. Reuse its database/model
mounts, environment, secrets, IAM roles, networking and task count. Publish an
immutable image tag and deploy a new task revision; do not recreate the stack.
Retain the old task revision/image for rollback. Wait for service stability and
healthy targets before considering deployment complete.

Verify through the production HTTPS address:
- Opening page has separate display and speaking language controls.
- Native selectors, links, alerts and primary controls use orange accents.
- Recording start/finish controls have visible localized text.
- Results, care choices and history work with an approved synthetic test account.

Local automated tests and a local theme preview are not proof of production
deployment. If AWS login or container-build access is unavailable, report that
blocker and leave the production state unchanged.
