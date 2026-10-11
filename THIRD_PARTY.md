# Third-party code and resources

Artisan includes code derived from [Articraft](https://github.com/mattzh72/articraft), distributed under the Apache License 2.0. The software license is included in `LICENSE`. Existing third-party notices are preserved in `NOTICE`, including the `cq_gears` implementation.

## Material textures

The bundled material library uses these ambientCG assets:

| Material | Source |
| --- | --- |
| wood | [Wood063](https://ambientcg.com/a/Wood063) |
| walnut_wood | [Wood068](https://ambientcg.com/a/Wood068) |
| fabric | [Fabric063](https://ambientcg.com/a/Fabric063) |
| brushed_metal | [Metal032](https://ambientcg.com/a/Metal032) |
| marble | [Marble016](https://ambientcg.com/a/Marble016) |
| concrete | [Concrete034](https://ambientcg.com/a/Concrete034) |
| carpet | [Carpet013](https://ambientcg.com/a/Carpet013) |

ambientCG publishes its downloadable assets under [CC0 1.0 Universal](https://docs.ambientcg.com/license/). The bundled file manifest records each texture's relative path and SHA-256 digest. The package includes the material files used by the renderer, so generation does not download textures implicitly.

## External executables and libraries

Blender is installed separately and is not bundled in the Python wheel. Python and frontend dependencies retain their own licenses; the dependency lock files identify the versions used for this release.

## Material evaluation

The material attribute evaluation and transfer implementation is adapted from
ARTVIP-Eval (copyright 2024 X-Humanoid), distributed under the Apache License 2.0.

## Geometry evaluation

The articulated geometry metrics follow the evaluation protocol released with
[Particulate](https://github.com/RuiningLi/particulate). Artisan implements these
metrics locally; the Particulate model is not required for inference or evaluation.
