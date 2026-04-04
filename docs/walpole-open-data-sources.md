# Walpole Open Data Sources

This note tracks the extra open-source material downloaded for the Walpole late-1700s material groups without silently mixing it into [data/reference](/Users/egjerde/code/dendrochronology/data/reference).

## What Was Added

The new files live under [data/walpole_open_sources/noaa](/Users/egjerde/code/dendrochronology/data/walpole_open_sources/noaa). The machine-readable inventory is [manifest.json](/Users/egjerde/code/dendrochronology/data/walpole_open_sources/manifest.json).

These are the highest-value additions from NOAA/NCEI:

| Material group | File | Why it matters |
| --- | --- | --- |
| `hemlock` | [vt013.rwl](/Users/egjerde/code/dendrochronology/data/walpole_open_sources/noaa/vt013.rwl) | Green River House historical hemlock in Vermont. This is the closest open Northeast historical-building analog found in this pass. |
| `oak` | [ma009.rwl](/Users/egjerde/code/dendrochronology/data/walpole_open_sources/noaa/ma009.rwl) | First Parish Church, Groton. Late-colonial New England structural oak series. |
| `oak` | [ny016.rwl](/Users/egjerde/code/dendrochronology/data/walpole_open_sources/noaa/ny016.rwl), [ny036.rwl](/Users/egjerde/code/dendrochronology/data/walpole_open_sources/noaa/ny036.rwl) | White-oak and red-oak regional references that expand the oak lane beyond a single church dataset. |
| `white_pine` | [ma022.rwl](/Users/egjerde/code/dendrochronology/data/walpole_open_sources/noaa/ma022.rwl), [me048.rwl](/Users/egjerde/code/dendrochronology/data/walpole_open_sources/noaa/me048.rwl), [ny043.rwl](/Users/egjerde/code/dendrochronology/data/walpole_open_sources/noaa/ny043.rwl) | Massachusetts, Maine, and New York white-pine references for exploratory species expansion. |
| `hard_pine` | [me037.rwl](/Users/egjerde/code/dendrochronology/data/walpole_open_sources/noaa/me037.rwl) | Open pitch-pine reference for the hard-pine caution lane. |
| `chestnut` | [tn012.rwl](/Users/egjerde/code/dendrochronology/data/walpole_open_sources/noaa/tn012.rwl), [tn013.rwl](/Users/egjerde/code/dendrochronology/data/walpole_open_sources/noaa/tn013.rwl), [tn012.crn](/Users/egjerde/code/dendrochronology/data/walpole_open_sources/noaa/tn012.crn), [tn013.crn](/Users/egjerde/code/dendrochronology/data/walpole_open_sources/noaa/tn013.crn) | Open American-chestnut reference material. These are outside New England, but they are better than shipping chestnut as a purely empty placeholder. |

## Important Distinction

These files are open ring-width tables and chronology files. They are not the same thing as benchmark-ready scan images.

That matters because the next Walpole benchmark we actually need is:

`scan -> measure -> infer-materials -> date`

The files in [data/walpole_open_sources/noaa](/Users/egjerde/code/dendrochronology/data/walpole_open_sources/noaa) help with reference expansion and species/material coverage, but they do not by themselves validate the imaging pipeline.

## Supplemental Material Info

These sources are useful context for why the material groups are plausible in Walpole-style late-1700s house construction:

- Historic New England, [Gilman Garrison House](https://www.historicnewengland.org/property/gilman-garrison-house/): describes a New Hampshire timber-framed house built with hemlock planks mortised into oak posts.
- Historic New England, [A to Z Primer for Homeowners](https://www.historicnewengland.org/preservation/for-homeowners-communities/your-old-or-historic-home/a-z-primer-for-homeowners/): notes long-running use of white or yellow pine boards in New England houses.
- University of New Hampshire Extension, [List of New Hampshire Native Trees](https://extension.unh.edu/resource/list-new-hampshire-native-trees-0): supports chestnut and oak regional plausibility.
- Harvard Forest, [Old-Growth Study Reconstructs Southern NH Forests](https://harvardforest.fas.harvard.edu/notes/old-growth-study-reconstructs-southern-nh-forests/): supports white-pine, hemlock, and oak as realistic southern New Hampshire materials.

## Gaps That Still Matter

- No benchmark-ready public scan-image corpus with known outer-ring years was found for white pine, oak, or chestnut in this pass.
- The chestnut files found here are Appalachian, not New England.
- The white-pine files found here are mostly 19th-20th century forest series, not late-1700s house-timber series.

So this sourcing pass improves the open reference pool, but it does not yet close the Walpole external-scan validation gap.
