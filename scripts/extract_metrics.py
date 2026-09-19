"""
Extract final training metrics from YOLO results.csv files.
"""

import csv
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

models = {
    'accident_detection': ROOT_DIR / "results" / "detection" / "yolo26_baseline" / "results.csv",
    'night_traffic': ROOT_DIR / "results" / "detection" / "night_traffic-2" / "results.csv",
    'accident_severity': ROOT_DIR / "results" / "detection" / "accident_severity-2" / "results.csv",
}

for model_name, csv_path in models.items():
    print(f"\n{'='*80}")
    print(f"Model: {model_name}")
    print(f"File: {csv_path}")
    print(f"{'='*80}")
    
    if not csv_path.exists():
        print(f"File not found!")
        continue
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        print(f"\nHeader: {header}")
        
        rows = list(reader)
        print(f"Total rows: {len(rows)}")
        
        # Get the last row (epoch 100)
        last_row = rows[-1]
        print(f"\nLast row (epoch {last_row[0]}):")
        print(f"  {last_row}")
        
        # Extract metrics
        if len(last_row) >= 8:
            epoch = last_row[0]
            precision = last_row[5]  # metrics/precision(B)
            recall = last_row[6]     # metrics/recall(B)
            map50 = last_row[7]     # metrics/mAP50(B)
            map50_95 = last_row[8]  # metrics/mAP50-95(B)
            
            print(f"\nMetrics:")
            print(f"  Precision (col 5): {precision}")
            print(f"  Recall (col 6): {recall}")
            print(f"  mAP@50 (col 7): {map50}")
            print(f"  mAP@50-95 (col 8): {map50_95}")
            
            # Convert to percentages
            try:
                precision_pct = float(precision) * 100
                recall_pct = float(recall) * 100
                map50_pct = float(map50) * 100
                map50_95_pct = float(map50_95) * 100
                
                print(f"\nMetrics (as percentages):")
                print(f"  Precision: {precision_pct:.2f}%")
                print(f"  Recall: {recall_pct:.2f}%")
                print(f"  mAP@50: {map50_pct:.2f}%")
                print(f"  mAP@50-95: {map50_95_pct:.2f}%")
            except ValueError as e:
                print(f"Error converting to float: {e}")
        else:
            print(f"Row has only {len(last_row)} columns, expected at least 8")
