# Semantic Repair Contract

Apply this contract to every HomeCheck finding before opening a family-specific
specification. The repair target is the program entity that owns the reported
behavior, not the diagnostic line in isolation.

## Required workflow

1. Resolve the owning entity: declaration, component, collection render, callback,
   layout subtree, resource/API call, or cryptographic protocol operation.
2. Collect every current finding on that entity and findings within five lines that
   belong to an interacting family.
3. Follow declarations, mutations, reads, call sites, imports, lifecycle hooks,
   resources, persisted formats, and tests until the invariants below are supported
   by repository evidence.
4. Choose one joint transformation for the entity. Repair all instances in every
   planned file; do not stop after a representative subset.
5. Build and run available tests before HomeCheck validation. When a gate fails,
   repair the retained candidate in place.
6. Treat a residual or introduced sibling finding as unfinished work in the next
   round. Never count an alert exchange as a successful repair.

## Invariants

Record the applicable invariants in the completion report:

- State ownership, mutation sources, reactive updates, and parent-child data flow.
- Lifecycle ordering and reuse/reset behavior.
- Callback parameter meaning and invocation timing.
- Collection-key stability and uniqueness, including duplicate values.
- Public and package-exported interfaces.
- Layout size, alignment, padding, clipping, hit testing, accessibility, and visual
  hierarchy.
- Execution order, asynchronous behavior, error propagation, and cancellation.
- Resource identity, protocol compatibility, persisted data, and cryptographic
  interoperability.
- Business literals and user-visible values.

## Completion standard

Risk changes how much evidence is required; it does not permit skipping a target.
Continue by expanding repository inspection, introducing a compatibility adapter,
or adding a narrowly scoped regression test when necessary. A target may be marked
`unresolved_external` only when a required fact is outside the repository and cannot
be derived from SDK documentation, call sites, configuration, or existing tests.
Such a target is not completed and must be reported separately.

For every rule/file group, report the owning entities, applicable invariants,
inspected evidence, transformation summary, and repaired alert locations. A generic
claim such as "followed the guide" is not evidence.

## Forbidden shortcuts

- Do not copy a before/after example as a patch. The 383 examples are provenance and
  may contain changed names, exports, decorators, literals, keys, salts, or behavior.
- Do not synthesize a collection index with `collection.indexOf(item)`.
- Do not add `@Reusable` without checking nested-component restrictions, lifecycle
  reset behavior, parameter kinds, and reuse identity.
- Do not remove or replace a state decorator without proving ownership, every
  mutation source, UI reads, and lifecycle updates.
- Do not delete commented text unless repository evidence shows it is dead code.
- Do not change business values, frame rates, collection contents, resource IDs,
  keys, IVs, salts, algorithms, stored formats, or protocol fields merely to match an
  example.
- Do not delete an enclosing component, function, method, class, or layout subtree to
  suppress a finding.
- Do not rename exported symbols or change public signatures unless the complete
  repository migration is intentional, justified, and accepted by the public-API
  gate.

