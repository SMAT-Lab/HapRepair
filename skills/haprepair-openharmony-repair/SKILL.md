---
name: haprepair-openharmony-repair
description: Repair all localized HomeCheck performance and security findings in OpenHarmony or ArkTS projects with the bundled HomeCheck controller, semantic-entity and interacting-rule-family planning, normative rule specifications, repository-wide evidence, build/test gates, residual/introduced-alert feedback, and namespace-export preservation. Use when Codex needs to scan, diagnose, repair, or validate CodeLinter/HomeCheck alerts without RAG or patch-template copying.
---

# HapRepair for OpenHarmony

Use the coding agent for repository inspection and edits. Use the bundled controller
for deterministic HomeCheck localization, complete alert planning, semantic guards,
build/tests, scan accounting, and candidate selection. Repair every localized target;
do not reduce the task to a safe-rule subset.

## Execution Roles And Source Safety

There are two explicit roles:

- In standalone use, the evaluator/operator runs `repair_session.py` and the agent
  edits the workspace between controller operations.
- In an evaluator-controlled agent turn, the evaluator has already initialized the
  session, supplied frozen findings and a plan, and owns every controller operation.
  The agent edits source/configuration files and writes the completion report only.
  It must not run `init-session`, `scan-initial`, `begin-round`,
  `record-completion`, `preflight-round`, `validate-round`, or `finalize`, and must
  not run HomeCheck scans or create a second plan. The evaluator's agent-mode guard
  rejects these operations.

Source preservation is the highest-priority invariant. Never delete or replace an
existing `struct`, `class`, component, function, method, `build()` method, exported
symbol, or public API, even when it appears unused. Do not replace a component by a
builder that removes the original declaration. Keep the declaration and signature;
edit its implementation or add a compatibility wrapper/builder alongside it. Do not
delete code, imports, fields, or containers merely to silence a checker. A structural
guard failure names the removed declarations; restore those declarations first and
retain all other useful edits.

## Import Graph And Package API Safety

For `@security/no-cycle`, inspect the complete directed import graph and the actual
uses on every edge before editing. Classify each edge as runtime, type-only, or
side-effect-bearing, and identify the package entry points and their baseline exports.
Break the cycle only with a behavior-preserving design: move genuinely shared
types/constants/side-effect-free helpers to the lowest common module, invert the
dependency through an existing interface, or convert an edge to `import type` only
when the mounted ArkTS compiler accepts it and the edge has no runtime use or side
effect. Update all affected imports and prove that initialization order and singleton
identity are unchanged.

Never delete an import or export, replace a module with `import * as`, copy a
definition into a second module, introduce a dynamic import, change a package entry
point, or rename an exported symbol merely to hide a cycle. A cycle repair that
touches `export *`, explicit re-exports, `@performance/hp-arkts-no-use-any-export-*`,
or `@performance/no-use-any-import` is one interaction family: preserve the initial
package API and repair all resulting import/export findings jointly. In particular,
preserve every baseline `export * as Namespace from '...'` declaration exactly;
named re-exports, object literals, and wrapper facades are not equivalent namespace
APIs. The evaluator's namespace-export guard treats removal or redirection of such
an export as a failed candidate. A public-API or namespace-export guard failure is
unfinished work, not a reason to repeat the same rewrite.

When a leaf module imports its own barrel (for example, a utility importing
`./utils/Index` while that barrel exports the leaf), break that cycle at the leaf:
replace the leaf's barrel import with a direct import from the defining module,
then keep the barrel's original `export *`, `export * as`, and named re-exports
unchanged. Do not replace a namespace export with an object literal, wrapper
facade, or copied function set; namespace identity and export semantics are part
of the public API even when current callers use only individual functions.

For interface-only cycles, first prove that every use on the selected edge is a
type position and has no module-initialization side effect. Use `import type` only
on those edges, verify that the mounted ArkTS compiler accepts it, and leave all
runtime imports and package-entry exports unchanged. If either proof is missing,
use a shared type/interface module or retain the alert with a concrete blocker.

Apply cycle edits in two passes: repair the smallest graph edge or leaf-barrel
edge, build and inspect the exported API, then address any residual export/`any`
family findings. Do not batch-rewrite every import in the strongly connected
component before observing the first gate result.

Before reporting completion, re-check the edited graph, package-entry exports,
compiler-supported import syntax, and the complete build gate. If no behavior-
preserving graph transformation is supported by the repository, keep the alert and
record the concrete dependency/API blocker rather than fabricating a replacement.

## Build, Resource, And Toolchain Safety

Treat the build gate as a validator, not as a reason to modify the build toolchain.
Do not edit `hvigorw`, `hvigor/hvigor-wrapper.js`, `.hvigor`, generated dependency
stores, SDK files, or evaluator-only environment files to bypass an `npmrc`,
`HVIGOR_USER_HOME`, cache-permission, `libGL`, or similar infrastructure error.
Report the command and error in the completion evidence and continue only with
project-source or project-configuration repairs. Compiler checker diagnostics such
as `any` are warning-class diagnostics under the evaluator's `ignoreWarning` hook;
module resolution, compilation, packaging, process, and resource-schema failures
remain fatal.

For `@performance/dark-color-mode-check`, inspect each affected module's existing
`base/element` resources, resource references, and module configuration before
editing. A dark-mode repair must use the supported resource schema and preserve the
application's explicit colors. Never create an empty `color.json`, empty `color`
array/object, or an unverified `colorMode` property merely to satisfy the checker;
the resulting resource file must be syntactically valid and non-empty. Reuse the
same named color keys and valid value forms as the module's base resources, and
verify the target API/schema against the mounted SDK or existing project examples.

## Verify And Initialize

Read [operations.md](references/operations.md), then verify the pinned tool and
references:

```bash
python3 <skill-dir>/scripts/homecheck.py verify
```

The Skill contains normative specifications for 71 rule identities: all 63 rules in
the frozen 383-pair corpus plus eight additional rules observed in the 35-project
input. The 383 examples remain non-normative provenance. Never copy an example as a
patch or use embedding search, similarity ranking, Top-k, a broker, or RAG.

Keep session state outside the project:

```bash
python3 <skill-dir>/scripts/repair_session.py init-session \
  --workspace <project-root> --state-dir <external-state-dir> \
  --max-validation-scans 5 \
  --build-command '<project build command>' \
  --test-command '<project test command>'
python3 <skill-dir>/scripts/repair_session.py scan-initial \
  --state-dir <external-state-dir>
python3 <skill-dir>/scripts/repair_session.py begin-round \
  --state-dir <external-state-dir>
```

Omit a gate only when it is genuinely unavailable. `begin-round` freezes every
current alert location and supplies candidate interaction clusters.

## Repair Every Frozen Target

1. Read `core_spec_path` and every family `spec_path` returned for each rule. Treat
   these specifications as normative. Open a corpus `guide_path` only as optional
   provenance and reject any example that changes unrelated behavior.
2. Resolve each alert to its owning semantic entity. Merge same-entity and supplied
   same-file interaction clusters before editing.
3. Inspect declarations, mutations, reads, call sites, imports, resources, lifecycle
   hooks, project configuration, persisted formats, and tests required by the
   applicable invariants. Use `rg` and targeted reads without a fixed context window.
4. Choose one joint behavior-preserving transformation for the entity. Complete all
   frozen locations in all planned files, including state, layout, lifecycle,
   collection-key, reuse, API, and security changes that require repository-wide
   edits. Risk determines evidence depth, not whether the target is attempted.
5. Never synthesize an index with `collection.indexOf(item)`. Never add `@Reusable`,
   change state decorators, alter security parameters, delete comments, or remove
   containers without the evidence required by the corresponding specification.
   Do not treat a resource-directory finding as permission to invent a resource
   schema: for dark-mode findings, make a valid non-empty theme resource from the
   module's actual base resources, or leave the existing resource configuration
   unchanged while recording the concrete schema/API blocker for evaluator review.
   For `@performance/hp-arkui-use-taskpool-for-web-request`, use the auditor-
   recognized TaskPool shape when the request work is safe to offload: keep request
   options, error handling, cancellation, response ordering, and `request.destroy()`
   behavior; put the eligible response computation in a module-scope `@Concurrent`
   function; make the network callback `async`; construct a `taskpool.Task` for that
   function and `await taskpool.execute(task)` inside the callback. Pass only
   transferable data to the worker and keep UI/state mutations on the caller side.
   A `.then(...)` continuation after `taskpool.execute(...)` can be semantically
   asynchronous but is not a reliable match for this checker, so convert it to the
   explicit async/await form when this preserves behavior. If extraction would
   violate serialization or lifecycle constraints, retain the alert and record the
   concrete blocker.
   For a barrel-cycle finding, prefer the direct-leaf-import transformation above;
   preserve namespace exports exactly and do not introduce object wrappers or
   copied definitions. For a package entry point, classify each exported name as
   a runtime value or a type before changing an import form; a type-only import is
   acceptable only when the entry-point export remains compiler- and API-valid.
6. Write the completion report described in [operations.md](references/operations.md).
   Report semantic entities, invariants, repository evidence, transformation, and
   every original alert location. Ordinary `blocked` entries are invalid.
7. In standalone use, run `record-completion`, `preflight-round`, and
   `validate-round`. In evaluator-controlled agent mode, stop after writing the
   completion report and let the evaluator run these operations. Repair any
   structural, semantic, public-API, build, or test failure in the retained
   candidate; these retries consume no HomeCheck scan.
8. After HomeCheck validation, begin another round whenever any finding remains.
   Repair residual and introduced findings jointly. A related rule disappearing while
   a sibling rule appears is unfinished and cannot promote the candidate to
   best-valid.

```bash
python3 <skill-dir>/scripts/repair_session.py record-completion \
  --state-dir <external-state-dir> --report <completion.json>
python3 <skill-dir>/scripts/repair_session.py preflight-round \
  --state-dir <external-state-dir>
python3 <skill-dir>/scripts/repair_session.py validate-round \
  --state-dir <external-state-dir>
```

Reuse the same agent thread for in-place gate repair within one active round. Start a
fresh thread after a HomeCheck validation scan and pass only the persisted plan,
feedback, and current candidate. Do not perform project-level or whole-round rollback.

## Finish

Finish when no target findings remain. If five validation scans are exhausted first,
report every residual as unresolved rather than claiming complete repair. A missing
external protocol or production key policy may be recorded as
`unresolved_external`, but it does not satisfy round completion and must not be
counted as repaired.

```bash
python3 <skill-dir>/scripts/repair_session.py finalize \
  --state-dir <external-state-dir>
```

The evaluator-only final scan is never fed back into editing. Report initial, final,
eliminated, remaining, introduced, and net-reduction counts; validation scans;
build/test availability and outcomes; interaction exchanges; unresolved targets; and
final candidate selection. HomeCheck alert elimination is not proof of semantic
correctness, real-project results concern detected alerts, and security conclusions
are limited to observed rules.

## Evaluation Integrity

Preserve every historical run and use a new versioned run ID after changing this
Skill or its evaluator. Never pool results across protocol versions. For held-out
correctness evaluation, exclude target/reference pairs and near duplicates. For a
coding-agent-only condition, do not expose this Skill, its specifications, corpus
examples, traces, or generated patches.
