# Frontend third-party notices

The distributed JavaScript and CSS include third-party components. Their full
license and notice texts are in [THIRD_PARTY_LICENSES.txt](public/THIRD_PARTY_LICENSES.txt).
The [package inventory](public/THIRD_PARTY_PACKAGES.json) records exact installed
versions and the source and digest of each notice.

The inventory follows modules with rendered code in the production Rollup chunks,
plus Tailwind CSS and Vite's injected runtime helpers. It includes transitive
dependencies, including React, Three.js, Radix UI, and the Markdown and syntax
highlighting libraries. Vite copies both files into `dist/` for binary distribution.

Where an npm package omits a license file, the notice includes its upstream
license text and source URL. For `format`, the original source and README
copyright notices are retained alongside the license published by its author.

Regenerate this inventory whenever the dependency lock or bundled module set
changes. Do not distribute a rebuilt bundle without its corresponding notices.
