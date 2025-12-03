# Static Timing Analysis (STA)

## Overview

Static timing analysis for SimpleEdgeAiSoC using OpenSTA.

## Quick Start

```bash
# Run STA analysis
./run_sta.sh

# View timing report
cat reports/timing_summary.rpt
```

## Requirements

- OpenSTA
- Liberty files (.lib)
- Synthesized netlist
- SDC constraints

## Reports Generated

- `timing_summary.rpt` - Overall timing summary
- `setup_violations.rpt` - Setup time violations
- `hold_violations.rpt` - Hold time violations
- `clock_report.rpt` - Clock analysis
