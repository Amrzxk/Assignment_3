#!/usr/bin/env python3
"""
Compare two training logs to identify differences in metrics, timing, and behavior.
Designed to compare logs/training_log_prev.txt (original run) with logs/training_log.txt (resumed run).
"""

import re
from typing import Dict, List, Tuple
from datetime import datetime
import sys

def parse_log_file(filepath: str) -> Dict:
    """Parse a training log file and extract key metrics."""
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    data = {
        'epochs': {},
        'start_time': None,
        'end_time': None,
        'total_duration': None,
        'device': None,
        'dataset_samples': None,
        'classes': None,
        'warnings': [],
        'errors': [],
        'resume_info': None,
    }
    
    current_epoch = None
    
    for line in lines:
        line = line.strip()
        
        # Extract timestamp
        ts_match = re.match(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
        if ts_match:
            timestamp = datetime.strptime(ts_match.group(1), '%Y-%m-%d %H:%M:%S')
            if data['start_time'] is None:
                data['start_time'] = timestamp
            data['end_time'] = timestamp
        
        # Device info
        if 'Using device:' in line:
            match = re.search(r'Using device: (.+)', line)
            if match:
                data['device'] = match.group(1)
        
        # Dataset info
        if 'Dataset pool size:' in line:
            match = re.search(r'Dataset pool size: (\d+) samples.*Planned training usage: (\d+)', line)
            if match:
                data['dataset_samples'] = {
                    'pool': int(match.group(1)),
                    'planned': int(match.group(2))
                }
        
        if 'Selected classes:' in line:
            match = re.search(r"Selected classes: \[([^\]]+)\]", line)
            if match:
                classes_str = match.group(1).replace("'", "").replace('"', '')
                data['classes'] = [c.strip() for c in classes_str.split(',')]
        
        # Resume info
        if 'Resuming from epoch' in line:
            match = re.search(r'Resuming from epoch (\d+)', line)
            if match:
                data['resume_info'] = {
                    'resumed_from': int(match.group(1)),
                    'timestamp': timestamp
                }
        
        # Epoch start
        if 'Starting epoch' in line:
            match = re.search(r'Starting epoch (\d+)/(\d+)', line)
            if match:
                current_epoch = int(match.group(1))
                data['epochs'][current_epoch] = {
                    'start_time': timestamp,
                    'samples_logs': [],
                    'summary': {}
                }
        
        # Sample progress logs
        if current_epoch and 'samples |' in line and 'Time for last' in line:
            match = re.search(r'Epoch (\d+): (\d+)/(\d+) samples.*Time for last (\d+) samples: ([0-9:]+).*time left.*: ([0-9:]+)', line, re.IGNORECASE)
            if match:
                data['epochs'][current_epoch]['samples_logs'].append({
                    'processed': int(match.group(2)),
                    'total': int(match.group(3)),
                    'window_time': match.group(5),
                    'time_left': match.group(6),
                    'timestamp': timestamp
                })
        
        # Epoch summary
        if current_epoch and 'Epoch [' in line and '] Summary:' in line:
            # Next few lines contain the summary
            pass
        
        if current_epoch and 'Avg Loss:' in line:
            match = re.search(r'Avg Loss: ([\d.]+).*Avg IoU: ([\d.]+).*LR: ([\d.]+)', line)
            if match:
                data['epochs'][current_epoch]['summary'] = {
                    'avg_loss': float(match.group(1)),
                    'avg_iou': float(match.group(2)),
                    'lr': float(match.group(3))
                }
        
        if current_epoch and 'Epoch Duration:' in line:
            match = re.search(r'Epoch Duration: ([0-9:]+)', line)
            if match:
                data['epochs'][current_epoch]['duration'] = match.group(1)
        
        # Warnings and errors
        if '- WARNING -' in line:
            data['warnings'].append(line)
        if '- ERROR -' in line:
            data['errors'].append(line)
        
        # Total training time
        if 'Training completed successfully in' in line:
            match = re.search(r'Training completed successfully in ([0-9:]+)', line)
            if match:
                data['total_duration'] = match.group(1)
    
    return data


def compare_epochs(prev_data: Dict, new_data: Dict) -> List[str]:
    """Compare epoch-level metrics between two runs."""
    
    differences = []
    
    prev_epochs = set(prev_data['epochs'].keys())
    new_epochs = set(new_data['epochs'].keys())
    
    if prev_epochs != new_epochs:
        differences.append(f"❌ **Epoch mismatch**: Previous run had epochs {sorted(prev_epochs)}, new run has {sorted(new_epochs)}")
    
    common_epochs = prev_epochs & new_epochs
    
    for epoch in sorted(common_epochs):
        prev_ep = prev_data['epochs'][epoch]
        new_ep = new_data['epochs'][epoch]
        
        # Compare summaries
        if 'summary' in prev_ep and 'summary' in new_ep:
            prev_sum = prev_ep['summary']
            new_sum = new_ep['summary']
            
            loss_diff = abs(prev_sum.get('avg_loss', 0) - new_sum.get('avg_loss', 0))
            iou_diff = abs(prev_sum.get('avg_iou', 0) - new_sum.get('avg_iou', 0))
            lr_diff = abs(prev_sum.get('avg_lr', 0) - new_sum.get('avg_lr', 0))
            
            if loss_diff > 0.001:
                differences.append(
                    f"⚠️  **Epoch {epoch} Loss difference**: "
                    f"Prev={prev_sum.get('avg_loss'):.4f}, New={new_sum.get('avg_loss'):.4f}, Δ={loss_diff:.4f}"
                )
            
            if iou_diff > 0.001:
                differences.append(
                    f"⚠️  **Epoch {epoch} IoU difference**: "
                    f"Prev={prev_sum.get('avg_iou'):.4f}, New={new_sum.get('avg_iou'):.4f}, Δ={iou_diff:.4f}"
                )
        
        # Compare durations
        if 'duration' in prev_ep and 'duration' in new_ep:
            if prev_ep['duration'] != new_ep['duration']:
                differences.append(
                    f"ℹ️  **Epoch {epoch} Duration**: Prev={prev_ep['duration']}, New={new_ep['duration']}"
                )
    
    return differences


def generate_report(prev_file: str, new_file: str, output_file: str):
    """Generate a markdown comparison report."""
    
    print(f"Parsing previous log: {prev_file}")
    prev_data = parse_log_file(prev_file)
    
    print(f"Parsing new log: {new_file}")
    new_data = parse_log_file(new_file)
    
    report_lines = [
        "# Training Log Comparison Report",
        "",
        "## Overview",
        "",
        f"- **Previous Run**: `{prev_file}`",
        f"- **New Run (Resumed)**: `{new_file}`",
        f"- **Comparison Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    
    # Resume info
    if new_data.get('resume_info'):
        resume = new_data['resume_info']
        report_lines.extend([
            "## ✅ Resume Information",
            "",
            f"- **Resumed from Epoch**: {resume['resumed_from']}",
            f"- **Resume Timestamp**: {resume['timestamp']}",
            "",
        ])
    else:
        report_lines.extend([
            "## ⚠️ Resume Information",
            "",
            "- No resume detected in new log (trained from scratch)",
            "",
        ])
    
    # Basic comparison
    report_lines.extend([
        "## Configuration Comparison",
        "",
        "| Parameter | Previous Run | New Run | Match |",
        "|-----------|--------------|---------|-------|",
    ])
    
    device_match = "✅" if prev_data.get('device') == new_data.get('device') else "❌"
    report_lines.append(f"| Device | {prev_data.get('device', 'N/A')} | {new_data.get('device', 'N/A')} | {device_match} |")
    
    classes_match = "✅" if prev_data.get('classes') == new_data.get('classes') else "❌"
    report_lines.append(f"| Classes | {prev_data.get('classes', 'N/A')} | {new_data.get('classes', 'N/A')} | {classes_match} |")
    
    if prev_data.get('dataset_samples') and new_data.get('dataset_samples'):
        ds_match = "✅" if prev_data['dataset_samples'] == new_data['dataset_samples'] else "❌"
        report_lines.append(
            f"| Dataset Samples | Pool={prev_data['dataset_samples']['pool']}, "
            f"Planned={prev_data['dataset_samples']['planned']} | "
            f"Pool={new_data['dataset_samples']['pool']}, "
            f"Planned={new_data['dataset_samples']['planned']} | {ds_match} |"
        )
    
    report_lines.extend(["", ""])
    
    # Epoch-by-epoch comparison
    report_lines.extend([
        "## Epoch-by-Epoch Metrics",
        "",
        "| Epoch | Prev Loss | New Loss | Prev IoU | New IoU | Prev LR | New LR |",
        "|-------|-----------|----------|----------|---------|---------|--------|",
    ])
    
    all_epochs = sorted(set(prev_data['epochs'].keys()) | set(new_data['epochs'].keys()))
    
    for epoch in all_epochs:
        prev_ep = prev_data['epochs'].get(epoch, {}).get('summary', {})
        new_ep = new_data['epochs'].get(epoch, {}).get('summary', {})
        
        prev_loss = f"{prev_ep.get('avg_loss', 0):.4f}" if prev_ep else "N/A"
        new_loss = f"{new_ep.get('avg_loss', 0):.4f}" if new_ep else "N/A"
        prev_iou = f"{prev_ep.get('avg_iou', 0):.4f}" if prev_ep else "N/A"
        new_iou = f"{new_ep.get('avg_iou', 0):.4f}" if new_ep else "N/A"
        prev_lr = f"{prev_ep.get('lr', 0):.6f}" if prev_ep else "N/A"
        new_lr = f"{new_ep.get('lr', 0):.6f}" if new_ep else "N/A"
        
        report_lines.append(
            f"| {epoch} | {prev_loss} | {new_loss} | {prev_iou} | {new_iou} | {prev_lr} | {new_lr} |"
        )
    
    report_lines.extend(["", ""])
    
    # Differences
    differences = compare_epochs(prev_data, new_data)
    
    if differences:
        report_lines.extend([
            "## ⚠️ Detected Differences",
            "",
        ])
        report_lines.extend(differences)
    else:
        report_lines.extend([
            "## ✅ No Significant Differences Detected",
            "",
            "The metrics between the previous run and the resumed run are identical or within acceptable tolerance.",
        ])
    
    report_lines.extend(["", ""])
    
    # Warnings
    if new_data['warnings']:
        report_lines.extend([
            "## Warnings in New Run",
            "",
        ])
        for warning in new_data['warnings'][:10]:  # Limit to first 10
            report_lines.append(f"- `{warning}`")
        if len(new_data['warnings']) > 10:
            report_lines.append(f"- ... and {len(new_data['warnings']) - 10} more warnings")
        report_lines.extend(["", ""])
    
    # Duration comparison
    report_lines.extend([
        "## Training Duration",
        "",
        f"- **Previous Run**: {prev_data.get('total_duration', 'N/A')}",
        f"- **New Run**: {new_data.get('total_duration', 'N/A')}",
        "",
    ])
    
    # Write report
    report_text = "\n".join(report_lines)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"Report generated: {output_file}")
    print("\nView the full report in: " + output_file)


if __name__ == "__main__":
    prev_log = "logs/training_log_prev.txt"
    new_log = "logs/training_log.txt"
    output = "log_comparison_report.md"
    
    if len(sys.argv) > 1:
        prev_log = sys.argv[1]
    if len(sys.argv) > 2:
        new_log = sys.argv[2]
    if len(sys.argv) > 3:
        output = sys.argv[3]
    
    generate_report(prev_log, new_log, output)

