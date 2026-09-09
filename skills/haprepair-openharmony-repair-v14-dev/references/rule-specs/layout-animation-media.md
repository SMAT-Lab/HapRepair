# Layout, Animation, And Media Rules

Use this specification for view-tree simplification, animation APIs, visibility,
media lifecycle, and presentation behavior.

## Layout

- `hp-arkui-remove-container-without-property` and
  `hp-arkui-remove-redundant-nest-container`: remove a container only after comparing
  its width/height propagation, alignment, padding, margin, clipping, z-order, hit
  testing, accessibility, conditional rendering, and child-count semantics. Move
  required attributes to the correct surviving node rather than dropping them.
- `hp-arkui-use-row-column-to-replace-flex`: select `Row` or `Column` from the actual
  direction and preserve wrapping, spacing, alignment, reverse order, and adaptive
  behavior. Keep `Flex` when semantics cannot be represented equivalently.
- `dark-color-mode-check`: use theme-aware resources while preserving explicit
  product colors and contrast requirements; do not replace literals blindly. Inspect
  each affected module's base resource keys and supported schema first. The dark
  resource must be syntactically valid and non-empty; never create an empty
  `color.json`, empty `color` object/array, or an unverified `colorMode` property just
  to suppress the finding. Keep base/dark key correspondence and use only value
  forms accepted by the mounted SDK or existing project resources.
- `hp-arkui-use-word-break-to-replace-zero-width-space`: remove inserted zero-width
  characters only after configuring equivalent `wordBreak` behavior and checking
  copy, search, accessibility, and localization output.

## Animation and interaction

- `hp-arkui-combine-same-arg-animateto`: combine adjacent calls only when duration,
  curve, delay, iteration, completion ordering, and affected state are equivalent.
- `hp-arkui-use-scale-to-replace-attr-animateto`: use transforms only when layout,
  hit testing, clipping, and final geometry remain equivalent.
- `hp-arkui-use-transition-to-replace-animateto`: preserve insertion/removal timing,
  state sequencing, completion callbacks, and interruption behavior.
- `hp-arkui-reduce-pangesture-distance`: use the runtime rule ID
  `hp-arkui-reduce-pangesture-distance`; choose a threshold from interaction intent
  and accessibility requirements rather than copying a literal.
- `tabs-on-change-check`: move the relevant state update from `Tabs.onChange` to
  `onAnimationStart` only when the intended timing is animation start. Preserve the
  selected index and any logic that must still run after selection completes.

## Media

- `hp-arkui-image-async-load`: preserve placeholder, error, sizing, and load-order
  behavior when moving decoding/loading off the synchronous path.
- `monitor-invisible-area-in-image-animation`: for API 17+, configure
  `monitorInvisibleArea` on `ImageAnimator` and preserve start/stop state across
  visibility changes.
- `lottie-animation-destroy-check`: release the exact animation instance at the
  corresponding lifecycle boundary without destroying shared or still-visible
  instances.
- `no-high-loaded-frame-rate-range`: retain the product's requested frame-rate
  contract. Use an adaptive or documented supported range; never change 120 to 60
  solely to silence the rule.
- `hp-arkui-suggest-use-effectkit-blur`: migrate to EffectKit only after checking API
  availability, blur radius/unit, edge behavior, fallback, rendering order, and
  visual equivalence.
