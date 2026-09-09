# Structural Security Rules

## `no-commented-code`

Classify each reported comment from repository evidence. Delete it only when it is
obsolete executable code with no documentation, migration, compatibility, example,
or operational purpose. Convert documentation-like content to the project's normal
documentation form. If the code records required behavior, restore and test it or
move it to versioned documentation; do not erase it merely to satisfy the checker.

## `no-cycle`

Find the complete import/dependency cycle. Move shared types, constants, or
side-effect-free utilities to the lowest common module, or invert the dependency
through an existing interface. For each edge, distinguish runtime use, type-only use,
and module side effects; `import type` is acceptable only when the mounted ArkTS
compiler supports it and the edge is genuinely type-only. Preserve initialization
order, module side effects, singleton identity, package-entry exports, and the
baseline public API. Update every import in the cycle and run the available
build/tests; deleting an import/export, replacing it with a wildcard import, copying
a definition, renaming an exported symbol, or changing a package barrel is not a
repair. Treat `export *`, explicit re-exports, and `no-use-any-import`/
`no-use-any-export-*` findings as the same interaction family and resolve them jointly.

When the cycle is a leaf-to-barrel cycle, prefer replacing the leaf's barrel
import with a direct import from the defining module. Preserve the barrel's
original `export *`, `export * as`, and named re-exports exactly. Replacing a
namespace export with an object literal, wrapper facade, or copied function set
changes namespace identity and is not behavior-preserving. For type-only edges,
prove the use is type-position-only and confirm `import type` compiles under the
mounted ArkTS toolchain before using it. Apply the smallest edge change first,
then build and inspect the package-entry API before addressing sibling export or
`any` findings.
