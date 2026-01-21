# Dendrochronology Dating Tool

A Python CLI tool for dating historic timber by cross-dating ring width measurements against ITRDB reference chronologies.

## Overview

This tool helps determine when trees were felled by:
1. Extracting ring width measurements from scanned wood cross-sections
2. Cross-dating against publicly available ITRDB chronologies
3. Identifying the calendar year of the outermost ring (felling year if bark edge present)

Designed for dating historic New England timber (1600s-1800s), but works for any region with ITRDB coverage.

## Installation

```bash
# Clone and enter directory
cd dendrochronology

# Create virtual environment and install
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

Requires Python 3.9+.

## Quick Start

```bash
# Activate environment
source venv/bin/activate

# Download reference chronologies (already done for northeastern US)
dendro download --states=me,nh,vt,ma,ct,ri,ny

# Check available references
dendro info

# Measure rings from a scan (interactive)
dendro measure sample.tiff --dpi=1200 --output=sample.csv

# Cross-date against references
dendro date sample.csv --era-start=1750 --era-end=1850 --bark-edge
```

## CLI Commands

### `dendro download`

Download reference chronologies from ITRDB (NOAA NCEI).

```bash
dendro download --states=me,nh,vt,ma --species=PIST,TSCA,QUAL,QURU
```

Options:
- `--states`, `-s`: State codes, comma-separated (default: me,nh,vt,ma,ct,ri,ny)
- `--species`, `-p`: Species codes, comma-separated (default: PIST,TSCA,QUAL,QURU)
- `--output`, `-o`: Output directory (default: data/reference)
- `--overwrite`: Re-download existing files

### `dendro info`

Show information about downloaded reference chronologies.

```bash
dendro info
```

### `dendro measure`

Extract ring widths from a scanned wood sample using an interactive viewer.

```bash
dendro measure sample.tiff --dpi=1200 --output=sample.csv
```

Options:
- `--dpi`, `-d`: Scanner resolution in DPI (default: 1200)
- `--output`, `-o`: Output file for measurements (CSV)

Interactive controls:
1. **Path mode**: Click to mark measurement path from bark (outer) to pith (center)
2. Press ENTER to switch to ring marking mode
3. **Ring mode**: Click to mark ring boundaries, or press 'A' for auto-detect
4. Press ENTER to save and close

### `dendro date`

Cross-date a sample against reference chronologies.

```bash
dendro date sample.csv --era-start=1750 --era-end=1850 --bark-edge
```

Options:
- `--reference`, `-r`: Reference directory (default: data/reference)
- `--era-start`: Earliest possible felling year (default: 1600)
- `--era-end`: Latest possible felling year (default: 1900)
- `--species`, `-p`: Filter by species codes
- `--states`, `-s`: Filter by state codes
- `--bark-edge/--no-bark-edge`: Sample includes bark edge (default: yes)
- `--output`, `-o`: Save results to JSON file
- `--top`, `-n`: Number of top matches to display (default: 10)

### `dendro parse`

Parse and display contents of a Tucson format file.

```bash
dendro parse data/reference/nh/nh001.rwl
```

## Sample Preparation

### Extraction Methods

**Option A: Cut Section (Recommended for exposed beam ends)**
- Cut a thin cross-section (1-2 cm thick) from beam end
- Provides full cross-section for scanning
- Best quality for ring measurement

**Option B: Increment Borer**
- 5mm diameter core from pith to bark
- Minimal damage, leaves small pluggable hole
- Core must be mounted and sanded

**Option C: Wedge Sample**
- V-cut into beam edge where not visible
- Good compromise between damage and quality

### Critical Requirements

1. **Include bark edge**: Sample MUST extend to the original outer surface (bark or waney edge) for exact felling year
2. **Include pith if possible**: The center helps verify ring count
3. **Clear radial path**: Need unobstructed path from near-center to bark
4. **Multiple samples**: Take 2-3 from different beams for cross-verification

### Surface Preparation

1. Sand progressively: 120 → 220 → 400 grit
2. For pine/hemlock: Wipe with mineral spirits to enhance contrast
3. Goal: Clearly visible ring boundaries (dark latewood vs light earlywood)

### Scanning

- **Minimum**: 1200 DPI (workable with good preparation)
- **Recommended**: 2400 DPI for narrow rings
- Use TIFF format (no compression artifacts)
- Scan in color, convert to grayscale in software
- Place flat sanded surface against scanner glass
- Use black backing behind sample

## How Cross-Dating Works

### Algorithm

1. **Detrend**: Remove age-related growth trend from ring width series using cubic spline or negative exponential curve
2. **Standardize**: Convert to dimensionless indices with mean ~1.0
3. **Slide**: Move sample series along reference chronology one year at a time
4. **Correlate**: Calculate Pearson correlation coefficient at each position
5. **Rank**: Find positions with highest correlation and t-value

### Confidence Metrics

| Metric | High Confidence | Medium | Low |
|--------|----------------|--------|-----|
| Correlation (r) | > 0.60 | 0.45-0.60 | < 0.45 |
| T-value | > 6.0 | 4.0-6.0 | < 4.0 |
| Overlap (years) | > 50 | 30-50 | < 30 |

### Sample Length Requirements

- **Minimum**: 30 rings (marginal confidence)
- **Recommended**: 50+ rings (good confidence)
- **Optimal**: 100+ rings (high confidence)

## Project Structure

```
dendrochronology/
├── pyproject.toml              # Dependencies and config
├── src/dendro/
│   ├── reference/              # ITRDB data handling
│   │   ├── tucson_parser.py    # Parse .rwl/.crn Tucson format
│   │   ├── downloader.py       # Download from NOAA NCEI
│   │   └── chronology_index.py # Index by species/location
│   ├── imaging/                # Ring width extraction
│   │   ├── ring_detector.py    # Detect ring boundaries
│   │   ├── path_sampler.py     # Sample along measurement path
│   │   └── viewer.py           # Interactive measurement UI
│   ├── crossdating/            # Dating algorithms
│   │   ├── detrend.py          # Growth trend removal
│   │   ├── correlator.py       # Sliding correlation
│   │   └── matcher.py          # Multi-chronology matching
│   └── cli/main.py             # CLI entry point
├── data/
│   └── reference/              # Downloaded ITRDB chronologies
└── tests/                      # Test suite (43 tests)
```

## Species Codes

Common northeastern US species:
- **PIST**: Pinus strobus (Eastern White Pine)
- **TSCA**: Tsuga canadensis (Eastern Hemlock)
- **QUAL**: Quercus alba (White Oak)
- **QURU**: Quercus rubra (Red Oak)
- **PCRU**: Picea rubens (Red Spruce)
- **THOC**: Thuja occidentalis (Northern White Cedar)

## Regional Marker Years

Notable climate events visible in ring patterns:
- **1816**: "Year Without Summer" - very narrow ring
- **1780s**: Series of severe winters
- **1790s**: Generally favorable growing conditions

## New Hampshire Reference Data

For dating historic NH timber, the most relevant downloaded references are:

| File | Species | Site | Cores | Years |
|------|---------|------|-------|-------|
| nh001 | Red Spruce (PCRU) | Nancy Brook | 30 | 1561-1972 |
| nh002 | Eastern Hemlock (TSCA) | Gibb's Brook | 25 | 1509-1981 |
| nh003 | Red Spruce (PCRU) | Nancy Brook | 31 | 1610-1979 |
| nh004 | Red Spruce (PCRU) | Mt. Washington | 35 | 1678-1976 |
| nh005 | Red Pine (PIRE) | Rattlesnake Mtn | 53 | 1690-2008 |

**128 tree cores** from NH cover the 1780-1800 period (relevant for late 1700s construction).

## Connecticut River Valley References (Walpole, NH Area)

For dating timber in the southern NH / VT border region (Walpole, Charlestown, Keene), these references are particularly valuable:

### Eastern Hemlock (TSCA) - Historic Buildings

| File | Site | Cores | Years | Notes |
|------|------|-------|-------|-------|
| vt010 | West Brattleboro Apartments | 10 | 1570-1869 | ~10 mi from Walpole |
| vt011 | West Dummerston Covered Bridge | 4 | 1557-1847 | Historic structure |
| vt013 | Green River House | 4 | 1568-1834 | Historic building |
| vt009 | Guilford Center Meeting House | 1 | 1453-1834 | Historic building |

### Natural Chronologies

| File | Species | Site | Cores | Years |
|------|---------|------|-------|-------|
| nh002 | Eastern Hemlock | Gibb's Brook | 25 | 1509-1981 |
| vt005 | Red Spruce | Bolton Mountain | 7 | 1668-2005 |
| ma017 | Eastern Hemlock | Cold River | 8 | 1650-2003 |
| ma018 | Eastern Hemlock | Alander Mountain | 6 | 1628-1987 |

**43+ hemlock cores** from the Connecticut River Valley region are ideal for dating late 1700s construction timber.

## Data Sources

Reference chronologies from ITRDB (International Tree-Ring Data Bank):
- https://www.ncei.noaa.gov/pub/data/paleo/treering/measurements/northamerica/usa/
- https://www.ncei.noaa.gov/pub/data/paleo/treering/chronologies/northamerica/usa/

## Running Tests

```bash
source venv/bin/activate
pytest tests/ -v
```

47 tests total, including 4 real-data validation tests using ITRDB chronologies.

## Algorithm Validation

The cross-dating algorithm has been validated against real ITRDB data:

| Test | Species | Site | Result |
|------|---------|------|--------|
| nh001 | Red Spruce (PCRU) | Nancy Brook, NH | Correct date (0 error) |
| nh002 | Eastern Hemlock (TSCA) | Gibb's Brook, NH | Correct date (0 error) |
| nh003 | Red Spruce (PCRU) | Nancy Brook, NH | Correct date (0 error) |
| ma015 | Mixed | Historic buildings, MA | Correct date (≤5 yr error) |

Run validation manually:
```bash
python scripts/validate_crossdating.py --num-tests=5
```

**Important**: Cross-dating works best when reference chronologies are from:
1. The same species as your sample
2. A geographically similar area (within ~100 miles)

## Example Output

```
CROSS-DATING RESULTS
============================================================
Sample: basement_beam_1
Length: 87 rings
Bark edge: Yes

PROPOSED FELLING YEAR: 1789
Confidence: HIGH

Top 10 matches:
------------------------------------------------------------
1. ma012 (MA)
   Felling year: 1789
   Correlation: 0.67, T-value: 7.2
   Overlap: 87 years, Confidence: HIGH

2. nh003 (NH)
   Felling year: 1789
   Correlation: 0.61, T-value: 6.1
   Overlap: 82 years, Confidence: HIGH
...
```

## License

MIT
