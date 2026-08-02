# Follow-up Items

This file records user-approved follow-up work for the Kerr spacetime FNO project.

Codex must not add, remove, reorder, start, cancel, complete, or change the
status of an item without explicit user approval.

The current task always has priority. A newly discovered idea must not interrupt
unfinished work.

## Workflow

When Codex discovers a possible improvement, bug, refactoring opportunity,
experiment idea, optimization, or research direction:

1. Do not implement it immediately.
2. Continue the current approved task.
3. Briefly report the idea to the user.
4. Explain:
   - what the idea is
   - why it may be useful
   - its expected benefit
   - its approximate scope
   - its possible risks
   - whether it affects the current task
5. Ask whether it should be recorded in this file.
6. Add it only after explicit user approval.
7. Do not begin it until the current task is complete and the user explicitly
   selects it as the next task.

## Status Definitions

- `Pending`: Approved for later consideration, but not yet selected.
- `Ready`: Approved and sufficiently defined to begin after the current task.
- `In Progress`: Explicitly selected by the user as the current task.
- `Blocked`: Cannot continue until a stated dependency or decision is resolved.
- `Completed`: Implemented, reviewed, and accepted by the user.
- `Cancelled`: Explicitly cancelled by the user.

## Priority Definitions

- `High`: Important to correctness, reproducibility, or a near-term project goal.
- `Medium`: Useful improvement without immediate urgency.
- `Low`: Optional cleanup, optimization, or exploratory idea.

Priority does not authorize Codex to start an item.

## Item Rules

Each item must:

- have a unique identifier
- describe one focused piece of work
- record why it was added
- state its expected scope
- state important risks or dependencies
- record the user's decision
- preserve its original meaning when its status changes

Codex must not silently merge unrelated items.

Codex must not split an item into additional work without first discussing it
with the user.

Completed and cancelled items should remain in this file unless the user
explicitly approves archival or removal.

## Active Items

No follow-up items have been approved yet.

## Item Template

<!--
Copy this template only after explicit user approval.

### FUP-001: Short English Title

- Status: Pending
- Priority: Medium
- Created: YYYY-MM-DD
- Source task:
- Motivation:
- Proposed scope:
- Expected benefit:
- Risks:
- Dependencies:
- Validation required:
- User decision:
- Notes:
-->
