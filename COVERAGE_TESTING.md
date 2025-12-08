# RCKangaroo Coverage Testing Framework

## Overview

This framework measures RCKangaroo's coverage quality by running multiple puzzle solves and analyzing:
- **K-factor variance** (how far from theoretical optimum)
- **Consistency** (coefficient of variation)
- **Speed stability**
- **Comparison to Quasi system**

## Quick Start

### 1. Ensure RCKangaroo is Built

```bash
# Switch to stable branch
git checkout claude/merge-kangaroo-builds-018upxMtUB2cDLk3Txgmetca

# Build with optimizations
make clean && make SM=86 USE_JACOBIAN=1 PROFILE=release -j
```

### 2. Run Coverage Test

```bash
# Run 20 puzzle 75 solves (takes ~10-20 minutes total)
./coverage_test.sh
```

**What it does:**
- Runs puzzle 75 twenty times (each run: 30-90 seconds)
- Extracts K-factor, solve time, speed from each run
- Calculates mean, std dev, min, max, coefficient of variation
- Saves results to `coverage_results/` directory

### 3. Analyze Results

```bash
# Find your results file
ls coverage_results/

# Analyze (replace with your actual filename)
python3 analyze_coverage.py coverage_results/rckangaroo_coverage_*_raw.txt
```

**Output includes:**
- Statistical analysis (mean, std dev, CV%)
- Comparison to Quasi system (K=1.0 baseline)
- Puzzle 135 time projections
- Recommendations based on results
- Visualization plots (if matplotlib available)

## Understanding the Results

### K-Factor Analysis

**What is K-factor?**
```
K = Actual Operations / Expected Operations
```

- **K = 1.0**: Theoretical best average (birthday paradox)
- **K < 1.0**: Lucky run (better than expected)
- **K > 1.0**: Unlucky run (worse than expected)

**Interpretation:**
- **K mean < 1.2**: Excellent coverage
- **K mean 1.2-1.5**: Normal variance (expected for pure random)
- **K mean > 1.5**: Poor coverage (needs improvement)

### Coefficient of Variation (CV%)

**What is CV%?**
```
CV% = (Standard Deviation / Mean) × 100
```

**Interpretation:**
- **CV% < 30%**: Consistent, predictable performance ✓
- **CV% 30-50%**: Moderate variance, acceptable ◐
- **CV% > 50%**: High unpredictability, problematic ✗

### Range Ratio

**What is range ratio?**
```
Range Ratio = Max Time / Min Time
```

**Interpretation:**
- **< 2x**: Very consistent
- **2-4x**: Normal variance
- **> 4x**: High variance (some runs MUCH slower)

## Example Results

### Good Coverage (Target)
```
K-Factor:
  Mean: 1.15
  CV%: 25%
  Range: 0.8 - 1.6 (2x)

Interpretation: Excellent! Close to Quasi performance.
```

### Poor Coverage (Current Concern)
```
K-Factor:
  Mean: 1.35
  CV%: 45%
  Range: 0.6 - 2.4 (4x)

Interpretation: High variance - quasi-random could help.
```

## Comparison to Quasi System

Your Quasi system reports:
- **K-factor**: 1.0 consistently
- **CV%**: ~10% (very low variance)
- **Range ratio**: ~1.2x (minimal variance)

**Goal**: Port Quasi approach to GPU while maintaining:
- K = 1.0±0.1
- CV% < 20%
- Speed > 12 GKeys/s

## Why This Matters for Puzzle 135

### Scenario 1: RCKangaroo (K=1.35, CV%=45%)
- **Mean time**: 440 years
- **Best case** (K=0.9): 293 years
- **Worst case** (K=2.0): 652 years
- **Problem**: 359-year spread! Can't plan resources.

### Scenario 2: Quasi-GPU (K=1.0, CV%=10%)
- **Mean time**: 436 years
- **Best case** (K=0.95): 414 years
- **Worst case** (K=1.05): 458 years
- **Advantage**: 44-year spread - predictable!

## Customizing Tests

### Test Different Puzzles

Edit `coverage_test.sh`:
```bash
TEST_RUNS=20        # Number of runs
PUZZLE=75           # Puzzle number (75 is fast, 85 for more data)
DP_BITS=14          # DP bits (14 for 75, 16 for 85)
```

**Recommendations:**
- **Puzzle 75**: Fast (~30-60s), good for quick testing
- **Puzzle 85**: Slower (~15-45min), better statistics
- **Puzzle 90**: Very slow (~4-12 hours), most accurate

### Test CPU vs GPU Balance

You can modify the script to test different CPU thread counts:
```bash
./rckangaroo ... -cpu 32   # Test with 32 CPU threads
./rckangaroo ... -cpu 64   # Test with 64 CPU threads
./rckangaroo ... -cpu 128  # Test with 128 CPU threads
```

## Next Steps

### If K-factor is Good (< 1.2)
1. ✓ RCKangaroo already has decent coverage
2. Still worth testing GPU quasi-random for consistency
3. Focus on scaling to more GPUs

### If K-factor is Poor (> 1.3)
1. ⚠️ Coverage improvement critical
2. GPU quasi-random implementation HIGH PRIORITY
3. Test hybrid approach: quasi-random jump selection

### GPU Quasi-Random Implementation

If results show high variance, next step is:
```bash
# Create new experimental branch
git checkout -b claude/gpu-quasi-random-hybrid

# Implement Halton-based jump selection
# Target: 12-13 GKeys/s with K=1.0
```

## Files in This Framework

- **coverage_test.sh**: Main test runner (bash)
- **analyze_coverage.py**: Statistical analysis (Python)
- **COVERAGE_TESTING.md**: This documentation
- **coverage_results/**: Output directory (created automatically)

## Troubleshooting

### "bc: command not found"
```bash
sudo apt-get install bc
```

### Python dependencies
```bash
pip3 install numpy matplotlib
# If matplotlib fails, analysis still works (just no plots)
```

### Low variance detected
If you get very low variance (CV% < 15%), it might indicate:
- Small sample size (run more tests)
- Puzzle too easy (try puzzle 85)
- Lucky streak (statistically possible with 20 runs)

### Test won't start
Check:
```bash
# Verify executable exists
ls -la ./rckangaroo

# Check it's the right branch
git branch

# Rebuild if needed
make clean && make SM=86 USE_JACOBIAN=1 PROFILE=release -j
```

## Advanced: Instrumented Coverage

For even deeper analysis, you can instrument `RCGpuCore.cu` to track:
- Magnitude distribution (how often each jump size used)
- Direction distribution (octant coverage)
- Collision patterns

This requires modifying the kernel to log jump statistics - documented separately.

## Questions?

Check the session history for:
- Why K=1.0 is the theoretical best
- Birthday paradox explanation
- Quasi vs random trade-offs
- GPU quasi-random implementation strategies
