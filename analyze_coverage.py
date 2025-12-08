#!/usr/bin/env python3
"""
RCKangaroo Coverage Analysis Tool
Analyzes coverage test results and compares to Quasi system
"""

import sys
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

class CoverageAnalyzer:
    def __init__(self, results_file):
        self.results_file = results_file
        self.runs = []
        self.k_factors = []
        self.solve_times = []
        self.speeds = []
        self.dp_counts = []

    def parse_results(self):
        """Parse raw results file"""
        with open(self.results_file, 'r') as f:
            for line in f:
                if line.startswith('Run_'):
                    match = re.search(r'time=([0-9:.]+).*speed=([0-9]+).*DPs=([0-9]+).*K=([0-9.]+)', line)
                    if match:
                        time_str = match.group(1)
                        # Convert time to seconds
                        time_parts = time_str.split(':')
                        if len(time_parts) == 3:
                            time_sec = int(time_parts[0]) * 3600 + int(time_parts[1]) * 60 + float(time_parts[2])
                        elif len(time_parts) == 2:
                            time_sec = int(time_parts[0]) * 60 + float(time_parts[1])
                        else:
                            time_sec = float(time_parts[0])

                        speed = int(match.group(2))
                        dp_count = int(match.group(3))
                        k_factor = float(match.group(4))

                        self.solve_times.append(time_sec)
                        self.speeds.append(speed)
                        self.dp_counts.append(dp_count)
                        self.k_factors.append(k_factor)

        print(f"✓ Parsed {len(self.k_factors)} successful runs")

    def calculate_statistics(self):
        """Calculate comprehensive statistics"""
        if not self.k_factors:
            print("Error: No data to analyze")
            return

        stats = {
            'k_factor': {
                'mean': np.mean(self.k_factors),
                'std': np.std(self.k_factors),
                'min': np.min(self.k_factors),
                'max': np.max(self.k_factors),
                'median': np.median(self.k_factors),
                'cv': (np.std(self.k_factors) / np.mean(self.k_factors)) * 100,
                'range_ratio': np.max(self.k_factors) / np.min(self.k_factors)
            },
            'solve_time': {
                'mean': np.mean(self.solve_times),
                'std': np.std(self.solve_times),
                'min': np.min(self.solve_times),
                'max': np.max(self.solve_times),
                'median': np.median(self.solve_times),
                'cv': (np.std(self.solve_times) / np.mean(self.solve_times)) * 100,
                'range_ratio': np.max(self.solve_times) / np.min(self.solve_times)
            },
            'speed': {
                'mean': np.mean(self.speeds),
                'std': np.std(self.speeds),
                'min': np.min(self.speeds),
                'max': np.max(self.speeds)
            }
        }

        return stats

    def print_report(self, stats):
        """Print detailed analysis report"""
        print("\n" + "="*70)
        print("          RCKangaroo Coverage Analysis Report")
        print("="*70)

        # K-Factor Analysis
        print("\n📊 K-FACTOR ANALYSIS")
        print("-" * 70)
        k = stats['k_factor']
        print(f"  Mean:                {k['mean']:.3f}")
        print(f"  Median:              {k['median']:.3f}")
        print(f"  Std Deviation:       {k['std']:.3f}")
        print(f"  Min (best luck):     {k['min']:.3f}")
        print(f"  Max (worst luck):    {k['max']:.3f}")
        print(f"  Range Ratio:         {k['range_ratio']:.2f}x (max/min)")
        print(f"  Coeff. Variation:    {k['cv']:.1f}%")

        # Interpretation
        print("\n  📋 Interpretation:")
        if k['mean'] < 1.2:
            print("     ✓ EXCELLENT - K-factor shows superior coverage")
        elif k['mean'] < 1.5:
            print("     ◐ GOOD - K-factor within expected range for random selection")
        else:
            print("     ✗ POOR - K-factor indicates coverage issues")

        if k['cv'] < 30:
            print("     ✓ CONSISTENT - Low variance in K-factors")
        elif k['cv'] < 50:
            print("     ◐ MODERATE - Acceptable variance")
        else:
            print("     ✗ HIGH VARIANCE - Unpredictable performance")

        # Solve Time Analysis
        print("\n⏱️  SOLVE TIME ANALYSIS")
        print("-" * 70)
        t = stats['solve_time']
        print(f"  Mean:                {t['mean']:.1f} seconds")
        print(f"  Median:              {t['median']:.1f} seconds")
        print(f"  Std Deviation:       {t['std']:.1f} seconds")
        print(f"  Min (fastest):       {t['min']:.1f} seconds")
        print(f"  Max (slowest):       {t['max']:.1f} seconds")
        print(f"  Range Ratio:         {t['range_ratio']:.2f}x (max/min)")
        print(f"  Coeff. Variation:    {t['cv']:.1f}%")

        # Speed Analysis
        print("\n⚡ SPEED ANALYSIS")
        print("-" * 70)
        s = stats['speed']
        print(f"  Mean:                {s['mean']:.0f} MKeys/s")
        print(f"  Std Deviation:       {s['std']:.0f} MKeys/s")
        print(f"  Min:                 {s['min']:.0f} MKeys/s")
        print(f"  Max:                 {s['max']:.0f} MKeys/s")

        # Comparison to Quasi
        print("\n🔬 COMPARISON TO QUASI SYSTEM")
        print("-" * 70)
        print("  Metric              │ RCKangaroo      │ Quasi System    │ Verdict")
        print("  " + "─"*66)
        print(f"  K-factor (mean)     │ {k['mean']:15.3f} │ {1.0:15.3f} │ ", end="")
        if k['mean'] <= 1.05:
            print("✓ RCKangaroo wins")
        elif k['mean'] <= 1.2:
            print("◐ Similar")
        else:
            print("✗ Quasi wins")

        print(f"  K-factor CV%        │ {k['cv']:14.1f}% │ {10.0:14.1f}% │ ", end="")
        if k['cv'] <= 15:
            print("✓ RCKangaroo wins")
        elif k['cv'] <= 35:
            print("◐ Similar")
        else:
            print("✗ Quasi wins")

        print(f"  Variance (range)    │ {k['range_ratio']:14.2f}x │ {1.2:14.2f}x │ ", end="")
        if k['range_ratio'] <= 1.5:
            print("✓ RCKangaroo wins")
        elif k['range_ratio'] <= 2.5:
            print("◐ Acceptable")
        else:
            print("✗ Quasi wins")

        # Efficiency Analysis
        print("\n💡 EFFECTIVE SPEED COMPARISON")
        print("-" * 70)
        rc_speed = 15500  # MKeys/s (from previous tests)
        rc_k = k['mean']
        rc_effective = rc_speed / rc_k

        quasi_speed = 1900  # MKeys/s
        quasi_k = 1.0
        quasi_effective = quasi_speed / quasi_k

        print(f"  RCKangaroo:     {rc_speed:.0f} MKeys/s ÷ K={rc_k:.2f} = {rc_effective:.0f} MKeys/s effective")
        print(f"  Quasi System:   {quasi_speed:.0f} MKeys/s ÷ K={quasi_k:.2f} = {quasi_effective:.0f} MKeys/s effective")
        print(f"\n  Speed Advantage: RCKangaroo is {rc_effective/quasi_effective:.1f}x faster (effective)")

        # Puzzle 135 Projection
        print("\n🎯 PUZZLE 135 PROJECTION (2 GPUs)")
        print("-" * 70)
        ops_135 = 2**68.17
        rc_time_years = ops_135 / (rc_effective * 1e6) / (365.25 * 24 * 3600)
        quasi_time_years = ops_135 / (quasi_effective * 1e6) / (365.25 * 24 * 3600)

        print(f"  RCKangaroo:     {rc_time_years:.0f} years (±{k['cv']:.0f}% variance)")
        print(f"  Quasi System:   {quasi_time_years:.0f} years (consistent)")
        print(f"\n  🔸 RCKangaroo faster BUT high variance means some runs could take {rc_time_years * k['range_ratio']:.0f}+ years!")
        print(f"  🔸 Quasi more predictable - critical for long-term planning")

        # Recommendations
        print("\n💭 RECOMMENDATIONS")
        print("-" * 70)
        if k['mean'] > 1.3 or k['cv'] > 40:
            print("  1. ⚠️  RCKangaroo shows high variance - coverage improvements needed")
            print("  2. 🎯 Consider hybrid approach: GPU speed + Quasi coverage")
            print("  3. 🧪 Test GPU quasi-random implementation (target: 12-13 GKeys/s @ K=1.0)")
            print("  4. 📊 If GPU+Quasi achieves K=1.0 @ 12 GK/s → SUPERIOR to current setup")
        elif k['mean'] < 1.2 and k['cv'] < 30:
            print("  1. ✓ RCKangaroo shows excellent coverage - current approach is solid")
            print("  2. 🔬 Still worth testing GPU quasi-random for consistency gains")
            print("  3. 🎯 For puzzle 135: consistency >> raw speed")
        else:
            print("  1. ◐ RCKangaroo shows acceptable coverage")
            print("  2. 🔬 GPU quasi-random worth exploring for high-value puzzles")
            print("  3. 🎯 Consistency becomes critical at puzzle 110+ scale")

        print("\n" + "="*70)

    def plot_distributions(self, output_dir):
        """Generate visualization plots"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('RCKangaroo Coverage Analysis', fontsize=16, fontweight='bold')

            # K-Factor Distribution
            ax = axes[0, 0]
            ax.hist(self.k_factors, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
            ax.axvline(np.mean(self.k_factors), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(self.k_factors):.3f}')
            ax.axvline(np.median(self.k_factors), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(self.k_factors):.3f}')
            ax.axvline(1.0, color='orange', linestyle=':', linewidth=2, label='Quasi: 1.0')
            ax.set_xlabel('K-Factor', fontweight='bold')
            ax.set_ylabel('Frequency', fontweight='bold')
            ax.set_title('K-Factor Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # K-Factor vs Run Number
            ax = axes[0, 1]
            ax.plot(range(1, len(self.k_factors)+1), self.k_factors, 'o-', color='steelblue', markersize=6)
            ax.axhline(np.mean(self.k_factors), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(self.k_factors):.3f}')
            ax.axhline(1.0, color='orange', linestyle=':', linewidth=2, label='Quasi: 1.0')
            ax.fill_between(range(1, len(self.k_factors)+1),
                           np.mean(self.k_factors) - np.std(self.k_factors),
                           np.mean(self.k_factors) + np.std(self.k_factors),
                           alpha=0.2, color='red', label='±1 StdDev')
            ax.set_xlabel('Run Number', fontweight='bold')
            ax.set_ylabel('K-Factor', fontweight='bold')
            ax.set_title('K-Factor Stability Over Runs')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Solve Time Distribution
            ax = axes[1, 0]
            ax.hist(self.solve_times, bins=20, color='forestgreen', edgecolor='black', alpha=0.7)
            ax.axvline(np.mean(self.solve_times), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(self.solve_times):.1f}s')
            ax.axvline(np.median(self.solve_times), color='blue', linestyle='--', linewidth=2, label=f'Median: {np.median(self.solve_times):.1f}s')
            ax.set_xlabel('Solve Time (seconds)', fontweight='bold')
            ax.set_ylabel('Frequency', fontweight='bold')
            ax.set_title('Solve Time Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # K-Factor vs Solve Time (correlation)
            ax = axes[1, 1]
            ax.scatter(self.k_factors, self.solve_times, c=range(len(self.k_factors)),
                      cmap='viridis', s=100, edgecolors='black', linewidths=1)
            # Add trend line
            z = np.polyfit(self.k_factors, self.solve_times, 1)
            p = np.poly1d(z)
            ax.plot(self.k_factors, p(self.k_factors), "r--", linewidth=2, alpha=0.8, label='Trend')
            ax.set_xlabel('K-Factor', fontweight='bold')
            ax.set_ylabel('Solve Time (seconds)', fontweight='bold')
            ax.set_title('K-Factor vs Solve Time Correlation')
            ax.legend()
            ax.grid(True, alpha=0.3)
            cbar = plt.colorbar(ax.collections[0], ax=ax)
            cbar.set_label('Run Sequence', fontweight='bold')

            plt.tight_layout()

            # Save plot
            plot_file = Path(output_dir) / f"coverage_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            print(f"\n📊 Plot saved to: {plot_file}")

            # Also try to display if possible
            # plt.show()  # Commented out for headless systems

        except Exception as e:
            print(f"Warning: Could not generate plots: {e}")
            print("(matplotlib may not be available or display not configured)")

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_coverage.py <results_file>")
        print("Example: python3 analyze_coverage.py coverage_results/rckangaroo_coverage_*_raw.txt")
        sys.exit(1)

    results_file = sys.argv[1]

    if not os.path.exists(results_file):
        print(f"Error: File not found: {results_file}")
        sys.exit(1)

    analyzer = CoverageAnalyzer(results_file)
    analyzer.parse_results()

    if not analyzer.k_factors:
        print("Error: No valid data found in results file")
        sys.exit(1)

    stats = analyzer.calculate_statistics()
    analyzer.print_report(stats)

    # Generate plots
    output_dir = os.path.dirname(results_file)
    if not output_dir:
        output_dir = "."
    analyzer.plot_distributions(output_dir)

if __name__ == '__main__':
    main()
