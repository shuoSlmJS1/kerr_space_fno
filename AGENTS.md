# AGENTS.md

## Project Overview

This repository contains the source code for the Kerr spacetime FNO research project.

The local repository is the code-development workspace used with Codex. A Remote SSH
Codex session works directly in the corresponding server repository. Their distinct
execution boundaries are defined in `Local and Remote Codex Roles` below.

The Linux research server provides:

- full dataset generation
- formal model training
- large-scale inference
- multi-GPU experiments
- long-running experiments
- large outputs
- model checkpoints
- experiment logs



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

## Project Authority and Scientific Records

The four long-lived research Markdown files are the project-semantic authority:

1. `FNO_KERR_EXPERIMENT_PLAN.md` is the highest-priority record of locked experiment
   protocols. Codex must not modify a locked protocol without explicit user approval.
2. `FNO_KERR_CURRENT_STATE.md` records the current research state, completed evidence,
   and active next step.
3. `SERVER_DATA_EXPERIMENT_REGISTRY.md` is the authoritative index of server datasets,
   checkpoints, outputs, and provenance. Planned assets and existing assets must always
   be distinguished.
4. `FNO_KERR_REASONING_LOG.md` records observations, candidate explanations,
   discriminating experiments, and changes in evidence strength. It must distinguish
   user decisions, measured experimental facts, and AI-assisted interpretation.

The active research stage is **Cross-model Benchmark Protocol v1**. Plan A, the FNO-only
Plan B bidirectional core, and the Plan B range through T4797 are complete. The only
current formal next step is `task-aligned model implementation + capacity matching`.
Do not start the Phase-I formal training runs without user approval.

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

## Local and Remote Codex Roles

### Local Codex Workspace

The local workspace is used for:

- reading and understanding code
- implementing code changes
- lightweight testing
- reviewing Git diffs
- preparing deployment manifests
- preparing code for later GitHub publication

Local Codex must not assume that formal server datasets or checkpoints are available
locally, must not treat synthetic or tiny smoke results as formal scientific evidence,
and must not initiate a remote connection or file transfer.

### Remote SSH Codex Workspace

An intentionally opened Remote SSH Codex session operates directly in:

`/home/shanjinshuo/fno_kerr/kerr_project`

For a user-approved project task/action, Remote SSH Codex may perform the applicable
operations below, subject to existing approval gates. Scientific runs additionally
require compliance with the locked protocol; protocol approval alone does not authorize
launching all planned experiments.

- read formal server data and provenance
- modify the server repository code
- run unit and regression tests
- use GPUs and `tmux`
- generate formal data, train models, run frozen evaluation, and resume defined workflows
- read and summarize formal JSON and result assets

Remote SSH Codex must not independently alter the scientific protocol, use server
credentials outside the established session, or perform remote Git publication.

### Research Server Runtime

- The research server has four NVIDIA GeForce RTX 4090 GPUs at host indices 0-3.
- The FNO-Kerr server Conda environment is `fno_srv`, located at
  `/home/shanjinshuo/miniconda3/envs/fno_srv`. Project Python/PyTorch workloads on
  the server must explicitly use this environment; do not assume the default shell
  has activated it. Normal activation and use for approved project work do not require
  maintenance approval; environment changes follow
  `System, Conda, Network, and SSH Boundaries`.
- This is a shared server. Before every authorized GPU workload, inspect current
  GPU visibility, memory usage, utilization, and compute processes. Never assume
  that an idle GPU is permanently allocated to this project, and never terminate,
  modify, or interfere with another user's processes.
  Follow `Shared Server Safety Boundary` for resource inspection and conflict avoidance.
- For formal experiment runs, record the selected host GPU and
  `CUDA_VISIBLE_DEVICES` mapping. Distinguish host GPU indices from process-local
  CUDA indices: selecting host GPU 1 alone makes it process-local `cuda:0`.
- A GPU query failing inside a restricted execution context is not sufficient
  evidence that the server GPU or driver is broken. Cross-check available read-only
  hardware and runtime evidence before drawing that conclusion.
- The historical depth, extended-epoch, Queue A, and width experiment scripts default
  to host GPU 1 unless `GPU_ID` is overridden. The remembered "GPU 0 primary /
  GPU 1-3 auxiliary" arrangement is not independently verified; it is a
  user-provided operating convention, not a confirmed allocation.
- Do not infer DDP/DataParallel support from the presence of multiple GPUs.
  Converting a single-GPU experiment to multi-GPU/DDP execution requires explicit
  user approval because it may affect benchmark semantics.

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

In Remote SSH mode, an explicitly approved FNO-Kerr workflow may also create its
authorized new artifacts in registered project-owned data/output locations, subject
to `Shared Server Safety Boundary` and all existing artifact/provenance protections.

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

Local synthetic, smoke, and temporary test inputs, outputs, datasets, logs, diagnostics,
checkpoints, and predictions must be deleted automatically after the test unless the user
explicitly requests their preservation. Permanent test code and fixtures remain tracked
project assets; formal server artifacts belong only in the server locations below.

## Operational Boundaries

### Local-mode Remote Boundary

When operating in the local workspace, Codex must never connect to the Linux research
server or another remote server; use SSH, SCP, SFTP, rsync, a remote shell, or remote
file transfer; execute remote commands; or upload, overwrite, move, delete, or modify
server files. Server actions from a local session are performed by the user or by a
separately opened Remote SSH Codex session.

### Remote SSH Protocol Boundary

In the intentionally opened Remote SSH workspace, distinguish these authorizations:

- **Scientific protocol approval** defines the experiment: architecture, data,
  training budget, normalization, metrics, evaluation, and related scientific choices.
  A locked protocol does not itself authorize launching every run it describes.
- **Project execution approval** authorizes a specific project task/action within the
  accepted task scope and, where applicable, the locked protocol: implementing benchmark
  code, running tests, launching a specifically approved training run, generating an
  approved artifact, or updating project documentation. Normal repository code/document
  work is project execution, not non-project maintenance. Existing low-risk permissions
  and action-specific approval gates continue to apply.
- **Maintenance approval** covers non-project/runtime maintenance: Codex runtime state,
  user-scoped permissions, environment maintenance, or explicitly identified user-owned
  runtime/configuration objects. It requires a specifically approved maintenance task,
  remains subject to the filesystem and system boundaries below, and does not authorize
  scientific protocol changes.

Scientific execution requires both project execution approval and compliance with the
locked protocol. Codex must hard-stop and report a protocol-impacting issue rather than
choosing a scientific change itself. Phase-I formal runs still require explicit approval.

Codex must never:

- store server passwords, server credentials, private keys, or access tokens
- add, delete, or modify Git remotes
- modify files outside this repository except for registered FNO-Kerr asset locations
  within an authorized workflow or narrowly identified paths under the current user's
  Codex/runtime state for an explicitly approved maintenance task; any additional path
  requires a separate task-specific exception satisfying every restriction in
  `Filesystem Write Scope`, not general permission to write in the user's home
- modify global system settings or global/system Git configuration
- modify any Conda environment other than `fno_srv`, including `base`, `fno_wave`,
  and other users' environments, or modify a global Python installation or system-level
  software configuration; the project environment `fno_srv` follows the specific maintenance
  requirements in `System, Conda, Network, and SSH Boundaries`
- modify unrelated projects

Codex must not automatically push, force-push, pull, fetch, clone, or otherwise publish
through Git. Any remote Git action requires separate explicit user approval and is not
part of ordinary Local or Remote SSH Codex operation.

Codex must not run destructive Git commands such as:

- `git reset --hard`
- `git clean -fd`
- `git clean -fdx`
- `git checkout -- .`
- `git restore .`
- forced branch deletion
- forced push
- history rewriting commands

### Shared Server Safety Boundary

#### Scope and Protocol Independence

These rules govern execution safety on the shared research server. They do not
authorize changes to model architecture, training budget, batch size, seed, dataset
definition, normalization, evaluation protocol, or Benchmark Protocol v1. Resource
shortages or safety constraints may cause a run to stop or wait; they must never
silently change the scientific protocol. Existing stricter scientific, Git, deletion,
overwrite, provenance, and destructive-operation rules remain in force. Approval
requirements below do not override an existing prohibition or grant administrator
authority.

#### Other Users' Files and Private Data

Remote Codex must not enter, enumerate, scan, read, copy, move, modify, or delete
another user's home directory, project directory, private storage, or private files,
even if Unix permissions allow access. System permission to read something does not
constitute project authorization to read it. Do not broadly search `/home` or unrelated
storage when known project paths are sufficient.

#### Other Users' Processes

Remote Codex must never kill, signal, suspend, attach to, debug, restart, renice, or
otherwise interfere with another user's process. It must not inspect that process's
memory, environment variables, file descriptors, or private command arguments.
For shared-resource conflict avoidance, inspect only the minimum normally visible,
unprivileged metadata necessary to identify resource ownership or GPU occupancy.
This exception does not authorize access to private process data or file contents.

#### GPU Conflict Avoidance

Before every authorized GPU workload:

1. Check current GPU visibility.
2. Inspect memory usage, utilization, and compute-process occupancy.
3. Determine ownership only to the minimum extent necessary.
4. Avoid GPUs already carrying another user's compute workload.
5. If availability or ownership is ambiguous, stop and report rather than guessing.

Never evict or interfere with another user's GPU workload. An idle GPU is not a
permanent allocation, and historical defaults such as `GPU_ID=1` do not constitute
allocation. Follow `Research Server Runtime` for host/process-local GPU mapping.
Do not convert a single-GPU run to DDP, DataParallel, or other multi-GPU execution
without explicit approval. Resource pressure must not silently change batch size,
training budget, or other locked benchmark settings.

#### Filesystem Write Scope

Normal autonomous write scope is limited to:

1. The FNO-Kerr repository for the currently authorized task.
2. Project-owned data/output locations explicitly registered for the authorized
   FNO-Kerr workflow.
3. Narrowly identified files under the current user's Codex/runtime state, only for
   an explicitly approved maintenance task.

Writing elsewhere is outside normal autonomous scope and requires an explicit reason
and explicit user approval of a separate task-specific exception identifying the exact
paths and operations.
Such an exception cannot override protections for other users, system configuration,
registered assets, destructive operations, Git, or scientific protocol, or any existing
absolute prohibition. It is not general permission to write elsewhere in the user's
home directory. An allowed location does not itself authorize file
modification, overwrite, deletion, cleanup, recursive changes, environment changes,
or Git operations; existing approval gates still apply. Registered project assets
retain all provenance and no-overwrite protections. Resolve symlinks when relevant
so that a repository path cannot be used to escape this boundary.

#### Destructive Operations

Do not perform broad or ambiguous destructive operations, including broad `rm -rf`,
filesystem-wide cleanup, unapproved `git clean`, or recursive `chmod`/`chown` outside
a narrowly verified target. Do not delete registered datasets, checkpoints, outputs,
or provenance records. Preserve all existing stricter Git and destructive-operation
rules; an approved path or task is not blanket cleanup permission.

#### System, Conda, Network, and SSH Boundaries

Unless an existing higher-priority rule explicitly permits the action and the user
has also given task-specific approval, Remote Codex must not:

- use `sudo`;
- modify `/etc`;
- modify systemd units or restart system services;
- modify NVIDIA drivers or global CUDA configuration;
- modify kernel/sysctl settings;
- modify firewall or routing;
- create or expose new listening network services, including loopback listeners;
- change server-wide proxy or SSH configuration;
- install or remove system packages;
- modify shared/global Python installations;
- modify another user's Conda environment;
- upgrade, rebuild, delete, or materially modify `fno_srv`;
- modify another user's SSH configuration or sockets;
- read SSH private keys, tokens, authentication payloads, or secrets.

User approval alone does not override an existing absolute prohibition or grant
administrator privileges.

`fno_srv` is the normal project execution environment. Activation and normal use for
approved project work are allowed without maintenance approval. Upgrading, rebuilding,
deleting, changing dependencies, or otherwise materially modifying it requires a
specifically approved environment-maintenance task, including any necessary narrowly
scoped filesystem exception. Such approval must still comply with all stricter
dependency and system rules; project execution approval alone does not authorize these
changes.

Existing stricter prohibitions on system/global changes, other users' private data,
and unrelated environments still apply. The existing user-scoped Remote SSH and proxy
mechanism may be used as already configured; do not silently convert it into a
system-wide configuration.

#### Minimum Necessary Inspection

Prefer known repository paths, registered project assets, and targeted metadata
queries. Do not broadly scan `/home`, `/tmp`, all process state, mounted storage, or
unrelated directories unless a specific approved task genuinely requires it. Any such
inspection must still respect the other-user privacy and process boundaries above.
Security or resource checks must return only information necessary for the current
decision and avoid exposing unrelated user information.

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

## Scientific Protocol Gate

Codex may autonomously resolve clear engineering issues that do not change experiment
semantics, including import, path, shape, logging, checkpoint-loading, and resume bugs.

Codex must stop and report before changing or deciding any of the following:

- dataset identities, splits, Q range, Q ordering, lambda grid, T, or step size
- Kerr physics, initial conditions, solver semantics, normalization, or target transform
- model architecture, capacity-matching rule, training budget, evaluation metric, or
  checkpoint-selection policy
- resolution set, addition or removal of a formal experiment, or protocol changes made
  in response to observed results

In particular, Codex must not respond to poor model performance by silently retuning,
rearchitecting, scaling a model, or adding experiments until it obtains a better result.

The default workflow is: `Protocol lock -> implementation -> unit tests -> tiny smoke ->
formal server workflow -> result summary -> scientific interpretation -> record update`.
Formal workflows should be unified and resumable, typically `generation -> qualification
-> training/evaluation -> summary`; reuse completed assets rather than rerunning them
without purpose.

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

## Artifact and Record Rules

Formal server datasets belong in `data/tasks/`. Formal checkpoints, metrics, predictions,
logs, and experiment summaries belong in `outputs/`. Do not overwrite a formal asset;
hard-stop on provenance mismatch, and never record a planned asset as existing.

A registry entry establishes asset identity, location, and provenance. It does not
itself authorize migration, overwrite, deletion, regeneration, or a change to the
canonical `data/tasks/` and `outputs/` conventions. Registered locations remain subject
to these conventions and all existing asset protections.

Update the four long-lived research records only when a protocol is locked or formally
changed, a formal dataset/checkpoint/output is generated, a formal stage completes, a
scientific interpretation changes, or the active next stage changes. Do not mechanically
update them for ordinary unit tests, tiny smoke tests, internal refactors, or engineering
bug fixes with no scientific meaning.

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

After local testing, clearly distinguish local validation from formal server validation
and automatically remove temporary local test artifacts unless the user explicitly asks
to preserve them. Retain approved project code and permanent tests.

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

Local Codex may prepare a deployment manifest but must not connect to or modify the
server. Remote SSH Codex may directly apply approved repository changes and run approved
server validation within a locked protocol; it must still not publish through remote Git
without explicit user approval.

## Git Policy

The user controls all commits and publication. Commit messages should directly describe
the research or engineering stage and must not use Conventional Commit prefixes such as
`feat:`, `fix:`, `docs:`, or `refactor:`. Before a commit, inspect `git status`, the
relevant diff, and `git diff --check`. Do not automatically push or force-push, and do not
independently rebase or rewrite history; any history rewrite requires explicit user
approval and an archive reference first.

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
- Perform only authorized Remote SSH repository/project operations or maintenance of
  narrowly identified paths under the current user's Codex/runtime state for an
  explicitly approved maintenance task. Any other maintenance target requires the
  separate task-specific exception defined in `Filesystem Write Scope`; server-level
  configuration changes remain prohibited under `Operational Boundaries`.
- Do not commit or publish without approval.
- Report actual validation honestly.
- Record only user-approved follow-up ideas.
