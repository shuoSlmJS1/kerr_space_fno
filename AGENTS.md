# AGENTS.md

## Project Overview

This repository contains the source code for the Kerr spacetime FNO research project.

The local repository is the code-development workspace used with Codex.

The Linux research server is used separately for:

- full dataset generation
- formal model training
- large-scale inference
- multi-GPU experiments
- long-running experiments
- large outputs
- model checkpoints
- experiment logs

Codex must operate only inside this local repository.

Codex must not directly operate the Linux research server.

## Primary Objective

Assist with developing, reviewing, testing, and documenting the Kerr FNO codebase.

The priorities are:

1. scientific and numerical correctness
2. reproducibility
3. preservation of existing experiment behavior
4. minimal and reviewable changes
5. a clean and maintainable working directory

Codex must not perform broad refactoring, unrelated optimization, or speculative
redesign unless explicitly requested.

## Repository Scope

Important repository content includes:

- `src/`: reusable project source code
- `scripts/`: data generation, training, inference, evaluation, and experiment scripts
- dependency and environment files in the repository root
- project documentation and configuration files

The local repository may not contain:

- full research datasets
- full experiment outputs
- large model checkpoints
- all server logs
- the complete server runtime environment

Codex must not assume that server-only data or resources exist locally.

## Local and Server Roles

### Local Codex Workspace

The local workspace is used for:

- reading and understanding code
- implementing code changes
- lightweight testing
- reviewing Git diffs
- preparing deployment manifests
- preparing code for later GitHub publication

### Linux Research Server

The Linux research server is used for:

- formal dataset generation
- full-scale training
- full-scale inference
- large experiment queues
- multi-GPU execution
- final server-side validation

The user performs all server operations manually.

## Language Rules

- All text printed to terminals must be in English.
- All text written to JSON files must be in English.
- All text written to logs must be in English.
- All text written to reports, result files, and generated output files must be
  in English.
- Code comments must be written in Chinese.
- Identifiers, variable names, function names, class names, module names, file
  names, and command-line arguments must use English.
- Explanations given directly to the user may be written in Chinese.
- Do not translate established mathematical, physical, or software identifiers
  into Chinese inside code.

## Encoding and Line-Ending Policy

All tracked text files must use UTF-8 encoding.

The repository standard is LF line endings.

This applies to:

- Python files
- shell scripts
- YAML files
- JSON files
- Markdown files
- plain-text files
- configuration files
- environment files
- requirements files
- Git-related text files

`.gitattributes` is the authoritative repository rule for line endings.

Codex must:

- create new tracked text files with LF line endings
- preserve LF when modifying existing tracked files
- avoid mixed line endings
- avoid introducing CRLF into source files
- avoid introducing CRLF into files executed on the Linux server
- run `git diff --check` after relevant changes
- inspect `git ls-files --eol` when a line-ending issue is suspected

Codex must not change global Git line-ending settings to solve a
repository-local problem.

## Working Directory Cleanliness

Codex must keep the local working directory clean and organized.

Codex must not leave unnecessary:

- temporary files
- backup files
- `.bak` files
- `.old` files
- numbered duplicate files
- scratch scripts
- debug scripts
- abandoned test files
- duplicated outputs
- temporary logs
- temporary datasets
- cache directories
- generated files that are no longer useful

Do not create backup copies unless the user explicitly requests them.

Git history is the normal recovery mechanism for tracked project files.

## File Deletion and Cleanup Rules

Explicit user approval is required before deleting:

- any Git-tracked file
- any file that existed before the current task
- any user-created file
- any existing project script
- any existing configuration file
- any existing project document
- any existing result selected for preservation
- any existing directory

Codex may delete temporary files that Codex itself created during the current
task when those files are no longer useful.

Codex should remove its own unnecessary temporary files rather than leaving them
in the repository.

### New File Boundaries

Creating a new file does not require prior approval only when all of the
following conditions are met:

- the file is inside the current project workspace
- the file is directly related to the current task
- the file does not overwrite an existing file
- the file name and purpose are clear
- the file is not an unnecessary backup, duplicate, numbered copy, or temporary
  artifact
- the new file is listed in the final task report

Task-specific temporary files may be created without prior approval, but they
must follow the existing cleanup rules. Editing-tool internal temporary files
that are automatically removed before the task ends are exempt from the ban on
parallel copies.

Codex must not use a new file to bypass the requirement to obtain approval
before modifying an existing file. Formal new files still require approval
before Git staging or committing. Unnecessary new files and temporary files
must be removed under the existing cleanup rules.

When an existing file is approved for modification, Codex must edit the original
path directly. Git history is the recovery mechanism; Codex must not create a
long-lived backup, replacement copy, numbered copy, or parallel version.

### Test Artifact Exception

Test artifacts must not be deleted immediately after a test.

For test inputs, outputs, temporary datasets, logs, and diagnostic files:

1. Keep them after the test finishes.
2. Report the actual test results to the user.
3. Allow the user to inspect or evaluate the results.
4. Wait for explicit user approval.
5. After approval, delete temporary test data and outputs that are no longer
   required.
6. Preserve only code, fixtures, test cases, or results that the user approves
   as part of the project.

## Operational Boundaries

Codex must never:

- connect to the Linux research server or another remote server
- use SSH, SCP, SFTP, rsync, a remote shell, or another remote file-transfer
  method
- execute commands on a remote server
- upload, overwrite, move, delete, or modify files on a remote server
- store server passwords, server credentials, private keys, or access tokens
- perform any remote Git operation, including `git pull`, `git push`,
  `git fetch`, `git clone`, or remote access to GitHub or another Git service
- add, delete, or modify Git remotes
- modify files outside this repository
- modify global system settings or global/system Git configuration
- modify the Conda `base` environment, the `fno_wave` environment, another
  Conda environment, a global Python installation, or system-level software
  configuration
- modify unrelated projects

Server upload, server execution, and GitHub pull and push are performed
manually by the user.

Codex must not run destructive Git commands such as:

- `git reset --hard`
- `git clean -fd`
- `git clean -fdx`
- `git checkout -- .`
- `git restore .`
- forced branch deletion
- forced push
- history rewriting commands

### Low-Risk Actions That Do Not Require Prior Approval

Codex may directly perform the following low-risk actions without asking the
user each time:

1. Read files and directories in the current project workspace.
2. Search code, inspect file contents, and analyze call relationships.
3. Run read-only commands in the current project workspace.
4. Create a new file that satisfies every condition in `New File Boundaries`.
5. Create temporary files needed for the current task, subject to the existing
   cleanup and test-artifact rules.
6. Run lightweight, read-only Git commands, including:
   - `git status`
   - `git status --short`
   - `git diff`
   - `git diff --stat`
   - `git diff --check`
   - `git log`
   - `git show`
   - `git branch`
   - `git ls-files`
   - `git ls-files --eol`
7. When Windows sandbox Git reports dubious ownership, use a command-scoped
   `safe.directory` override for this repository only with the read-only Git
   commands listed above.
8. Run lightweight local checks that the user has already explicitly approved,
   while following the existing resource, scope, and test-artifact rules.
9. Generate deployment manifests, change manifests, publication checklists,
   candidate implementation plans, and read-only analysis reports, without
   deploying, uploading, pulling, pushing, or publishing.

## Current Task Focus

Codex must remain focused on the current user-approved task.

Codex must not interrupt an unfinished task to implement:

- unrelated improvements
- unrelated refactoring
- new experiments
- speculative optimizations
- additional features
- newly discovered research directions

If a new idea appears while the current task is unfinished, Codex must not
implement it immediately.

## Follow-up Idea Policy

The project follow-up file is:

`FOLLOW_UPS.md`

If Codex discovers an unrelated improvement, experiment idea, refactoring
opportunity, optimization, bug, or research direction:

1. Do not implement it.
2. Do not interrupt the current task.
3. Briefly report the idea to the user.
4. Explain:
   - what the idea is
   - why it may be useful
   - its expected benefit
   - its approximate scope
   - its possible risks
   - whether it affects the current task
5. Ask whether the idea should be added to `FOLLOW_UPS.md`.
6. Add it only after explicit user approval.

Codex must not silently:

- add follow-up items
- remove follow-up items
- reorder follow-up items
- change follow-up status
- begin a follow-up item

Codex may begin a follow-up item only when:

- the current task is complete, and
- the user explicitly selects that item as the next task

The Codex application conversation history must not be treated as the permanent
project follow-up system.

Approved follow-up items must be stored in `FOLLOW_UPS.md`.

## Change Procedure

Low-risk actions listed in `Operational Boundaries` may be performed directly.
Before modifying an existing file, Codex must have explicit approval for that
specific modification. For every approved coding task:

1. Read the relevant existing files.
2. Understand the current behavior and interfaces.
3. Identify the smallest practical set of files that must change.
4. Briefly explain the intended change.
5. Distinguish whether the task is:
   - a bug fix
   - a refactor
   - an experiment change
   - a feature
   - a performance optimization
   - a documentation change
6. Preserve existing interfaces and experiment behavior unless the task
   explicitly requires changing them.
7. Make the smallest practical implementation.
8. Run appropriate lightweight local checks.
9. Review the final Git diff.
10. Report the files changed, validation performed, and remaining risks.

Ordinary low-risk analysis and new-file creation may proceed without repeated
approval when they satisfy the stated boundaries. Existing-file modifications
and all other approval-gated actions require explicit user approval before
execution.

## Approval Gates

Before performing any of the following, Codex must explain the specific plan and
obtain explicit user approval:

1. Modify the contents of any file that existed before the current task,
   including any Git-tracked file.
2. Overwrite any existing file.
3. Delete, rename, or move any existing file or directory.
4. Modify multiple existing files through bulk formatting, refactoring,
   standardized rewriting, or bulk migration.
5. Perform any local Git write operation, including:
   - `git add`
   - `git commit`
   - `git merge`
   - `git rebase`
   - `git cherry-pick`
   - `git revert`
   - `git reset`
   - `git restore`
   - `git checkout` that changes the worktree, index, or branch state
   - `git switch` that changes branch state
   - branch creation, deletion, or renaming
   - tag creation or deletion
   - `git stash`
   - `git clean`
   - `git rm`
   - `git mv`
6. Install, remove, upgrade, or downgrade a dependency.
7. Modify Python, PyTorch, CUDA, or another package configuration in
   `fno_codex_local`.
8. Run a local test that exceeds lightweight local validation but is not
   absolutely prohibited.

This includes any proposed change to a public interface, command-line
interface, dataset format, checkpoint format, model input or output shape,
tensor layout, normalization behavior, target transformation, evaluation metric,
dataset split, random-seed policy, experiment naming convention, formal
experiment definition, numerical tolerance, physical constant, solver setting,
or integration setting.

### Current-Message Approval

When the user explicitly requests a modification to an existing file in the
current message, that request is approval for that specified modification in
this task. Codex must not ask again for approval for the same already-approved
modification.

Codex must explain the reason and obtain new approval if implementation requires
an additional existing file, an expanded scope, deletion, moving or renaming a
file, a public-interface change, a dataset-format change, a model-structure
change, a training-flow change, or an experiment-definition change.

Codex may generate a deployment manifest or publication checklist without prior
approval. Generating a manifest is not deployment.

## Research-Code Rules

Codex must preserve scientific reproducibility.

Codex must not silently change:

- random seeds
- training, validation, or test splits
- evaluation metrics
- normalization
- target transformations
- model dimensions
- tensor dimensions
- checkpoint structure
- experiment names
- dataset naming rules
- numerical tolerances
- physical constants
- solver settings
- integration settings

Codex must clearly distinguish:

- bug fixes
- code cleanup
- refactoring
- experimental changes
- numerical-method changes
- performance optimization

Do not replace a scientific or numerical method merely because another
implementation is shorter, newer, or more fashionable.

Mathematical, physical, and numerical assumptions should be documented near the
relevant implementation.

When uncertain about:

- Kerr spacetime physics
- geodesic equations
- turning-point logic
- numerical integration
- solver validity
- FNO architecture
- tensor dimensions
- normalization
- experiment definitions

Codex must stop and ask the user rather than guessing.

## Local Conda Environment Policy

The dedicated local Codex Conda environment is:

`fno_codex_local`

Codex must use this environment for local Python execution and testing.

Codex must not modify:

- the Conda `base` environment
- any other local Conda environment
- any global Python installation
- any system-wide package installation
- the Linux server environment

Codex may inspect the active environment and use packages that are already
installed in `fno_codex_local`.

Explicit user approval is required before:

- installing a package
- removing a package
- upgrading a package
- downgrading a package
- changing Python
- changing PyTorch
- changing CUDA-related packages
- changing NumPy
- changing SciPy
- changing another core package
- modifying an environment YAML file
- modifying a requirements file
- recreating the environment

Before proposing a dependency change, Codex must explain:

- why the dependency is needed
- whether the Python standard library can replace it
- whether an existing project dependency can replace it
- compatibility risks on Windows
- compatibility risks on the Linux server
- whether the dependency is required locally
- whether the dependency is required on the server
- whether the dependency is required in both environments

Codex must not use Docker unless the user explicitly decides to introduce
Docker into the project.

## Local Testing Environment

### Local Machine

- Operating system: Windows
- GPU: NVIDIA RTX 4060 Laptop GPU
- Purpose: lightweight development and validation

### Research Server

- Operating system: Linux
- GPU: 4 NVIDIA RTX 4090 GPUs
- Purpose:
  - formal dataset generation
  - full training
  - large-scale inference
  - multi-GPU experiments
  - long-running experiments
  - final server-side validation

The local computer has useful GPU capability, but it is not equivalent to the
server environment.

## Local Testing Rules

Codex may perform lightweight local tests, including:

- Python syntax checks
- AST parsing checks
- import checks
- small synthetic-input tests
- tensor-shape checks
- small forward-pass tests
- lightweight data-loader checks
- CPU smoke tests
- single-GPU smoke tests
- a very small number of training iterations
- a very small number of training epochs
- small-data numerical checks
- `git diff --check`
- line-ending checks

Codex must not start a large, expensive, or long-running local experiment
without explicit user approval.

Codex must not:

- download a large dataset without approval
- generate a full formal dataset locally without approval
- run full-scale training locally without approval
- run a long experiment queue locally without approval
- occupy the local GPU for a long time without approval
- represent local smoke-test results as formal server results

Before a nontrivial local test, Codex must briefly state:

- what will be tested
- what data will be used
- whether CPU or GPU will be used
- the approximate expected resource cost
- the approximate expected duration
- which temporary files may be created

After testing:

1. Keep temporary test inputs, outputs, logs, and diagnostic files.
2. Report the actual results to the user.
3. Clearly distinguish local validation from formal server validation.
4. Wait for the user to evaluate and approve the result.
5. After approval, retain approved project code and approved permanent tests.
6. Remove temporary test data, logs, outputs, and diagnostics that are no longer
   needed.

Codex must never claim that the following passed unless they were actually run
in the corresponding environment:

- full training
- full inference
- complete dataset validation
- multi-GPU execution
- Linux server validation
- formal experiment reproduction

## Dependency Rules

Reuse existing dependencies whenever practical.

Do not install a package merely because it makes an implementation shorter.

When a new dependency is proposed, explain:

- why it is needed
- what existing alternatives were considered
- whether it affects only local testing
- whether it affects the Linux server
- whether environment files must change
- whether reproducibility is affected

Do not automatically modify the Conda environment.

## Deployment Manifest Policy

Codex may generate a deployment manifest without prior approval.

The deployment manifest should separate files into:

### Modified Files

Files that already exist on the server and should be overwritten.

### Added Files

New files that should be uploaded to the corresponding server path.

### Deleted Files

Tracked or existing files proposed for removal.

The user must explicitly approve the deletion list and manually perform the
server-side deletion.

### Server Validation

Commands that the user may manually run on the server after deployment.

### Environment Attention

Dependency, environment, path, or compatibility changes that require manual
review.

Codex must not:

- connect to the server
- upload files
- overwrite server files
- remove server files
- execute server commands

All deployment actions are performed manually by the user.

## Git Policy

The user controls all commits and all remote publication. Remote Git operations
are prohibited and must be performed manually by the user.

Codex may directly run the lightweight, read-only Git commands listed in
`Operational Boundaries`. When Git reports dubious ownership inside the Windows
sandbox, Codex may use a command-scoped `safe.directory` override for this
repository only with those read-only commands. Codex must not modify global or
system Git configuration.

After making changes, Codex must:

1. Show or summarize `git status --short`.
2. Show or summarize the relevant `git diff`.
3. List modified files.
4. List added files.
5. List files proposed for deletion.
6. State which validation checks were actually run.
7. State which validation checks were not run.

All local Git write operations require explicit approval under `Approval Gates`.
Codex must not stage, commit, alter branches or tags, or otherwise change local
Git state without that approval.

A commit may be created only after the user explicitly approves the exact
change set.

## GitHub Publication Policy

The local Codex workspace is the basis for later GitHub publication.

Codex may prepare:

- a publication checklist
- a clean repository review
- a README draft
- an environment review
- a release manifest
- a list of files that should not be published
- a list of missing reproducibility materials

Codex must not push to GitHub.

Codex must not store GitHub credentials.

The user performs all GitHub operations manually.

Large datasets, full experiment outputs, large checkpoints, caches, and
temporary files should not be added to GitHub unless the user explicitly
decides otherwise.

## Task Reporting

For small tasks, provide a concise report containing:

- files changed
- checks performed
- remaining risks

For larger tasks, use:

### Summary

What was changed.

### Files

Modified, added, and proposed deleted files.

### Validation

Commands and checks that were actually run.

### Risks

Anything that still requires user review or server-side validation.

### Next Action

Exactly one recommended next step.

Do not claim that a check passed unless it was actually executed.

Do not hide incomplete validation.

## Final Behavior Principles

- Stay within the current task.
- Prefer minimal changes.
- Keep the repository clean.
- Preserve reproducibility.
- Ask before high-impact changes.
- Do not guess about scientific assumptions.
- Do not modify remote systems.
- Do not commit or publish without approval.
- Report actual validation honestly.
- Record only user-approved follow-up ideas.
