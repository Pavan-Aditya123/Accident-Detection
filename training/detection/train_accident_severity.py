from pathlib import Path
import torch
from ultralytics import YOLO


def main():

    # Project root directory
    base_dir = Path(__file__).resolve().parent.parent.parent

    # Dataset path
    data_yaml = base_dir / "datasets" / "accident_severity" / "data.yaml"

    # Results directory
    project_dir = base_dir / "results" / "detection"

    print("=" * 60)
    print("       YOLO26 ACCIDENT SEVERITY TRAINING")
    print("=" * 60)

    print(f"[+] Repository Root   : {base_dir}")
    print(f"[+] Dataset Config    : {data_yaml}")
    print(f"[+] Output Directory  : {project_dir / 'accident_severity'}")


    # Check dataset
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_yaml}"
        )


    # GPU check
    if torch.cuda.is_available():

        device = 0
        gpu_name = torch.cuda.get_device_name(0)

        print(f"[+] Compute Device    : GPU 0 ({gpu_name})")

    else:

        device = "cpu"
        print("[!] Compute Device    : CPU")


    # Load YOLO26n pretrained model
    model_name = "yolo26n.pt"

    print(f"[+] Loading Model     : {model_name}")

    model = YOLO(model_name)


    print("\n[+] Starting Training...\n")


    results = model.train(

        # Dataset
        data=str(data_yaml),

        # Training parameters
        epochs=100,
        imgsz=640,
        batch=8,

        # Hardware
        device=device,
        workers=4,

        # Optimization
        optimizer="AdamW",

        # Early stopping
        patience=20,

        # Saving
        save=True,
        save_period=10,

        # Validation and plots
        val=True,
        plots=True,

        # Experiment output
        project=str(project_dir),
        name="accident_severity",

        # Prevent overwriting previous experiments
        exist_ok=False
    )


    print("\n" + "=" * 60)
    print("       TRAINING COMPLETE")
    print("=" * 60)

    output_dir = project_dir / "accident_severity"

    print(f"[+] Best Model : {output_dir / 'weights' / 'best.pt'}")
    print(f"[+] Last Model : {output_dir / 'weights' / 'last.pt'}")
    print(f"[+] Results    : {output_dir}")


    # ── Post-training evaluation ───────────────────────────────────────────────

    best_weights = output_dir / "weights" / "best.pt"

    if not best_weights.exists():
        print(f"\n[!] best.pt not found at {best_weights} — skipping evaluation.")
        return

    print("\n" + "=" * 60)
    print("       POST-TRAINING EVALUATION")
    print("=" * 60)

    eval_model = YOLO(str(best_weights))

    CLASS_NAMES = ["moderate-accident", "no-accident", "severe-accident"]


    # Validation set
    print("\n[+] Evaluating on Validation Set...")

    val_metrics = eval_model.val(
        data=str(data_yaml),
        split="val",
        imgsz=640,
        device=device,
        verbose=True,
    )


    # Test set
    print("\n[+] Evaluating on Test Set...")

    test_metrics = eval_model.val(
        data=str(data_yaml),
        split="test",
        imgsz=640,
        device=device,
        verbose=True,
    )


    # ── Training summary ───────────────────────────────────────────────────────

    summary_path = output_dir / "training_summary.txt"

    with open(summary_path, "w") as f:

        f.write("=" * 60 + "\n")
        f.write("  ACCIDENT SEVERITY MODEL — TRAINING SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        f.write("MODEL\n")
        f.write(f"  Architecture : YOLO26n\n")
        f.write(f"  Weights      : {best_weights}\n\n")

        f.write("DATASET\n")
        f.write(f"  Path         : {data_yaml}\n")
        f.write(f"  Classes      : {len(CLASS_NAMES)}\n")
        f.write(f"  Class Names  : {', '.join(CLASS_NAMES)}\n\n")

        f.write("TRAINING CONFIGURATION\n")
        f.write(f"  Epochs       : 100\n")
        f.write(f"  Image Size   : 640\n")
        f.write(f"  Batch Size   : 8\n")
        f.write(f"  Optimizer    : AdamW\n")
        f.write(f"  Patience     : 20\n")
        f.write(f"  Device       : {device}\n\n")

        # Best epoch from results object
        try:
            best_epoch = int(results.best_epoch) + 1
        except Exception:
            best_epoch = "N/A"

        f.write(f"BEST EPOCH    : {best_epoch}\n\n")

        # Validation metrics
        f.write("VALIDATION METRICS\n")
        try:
            f.write(f"  Precision    : {val_metrics.box.mp:.4f}\n")
            f.write(f"  Recall       : {val_metrics.box.mr:.4f}\n")
            f.write(f"  mAP@50       : {val_metrics.box.map50:.4f}\n")
            f.write(f"  mAP@50-95    : {val_metrics.box.map:.4f}\n\n")

            f.write("  Per-Class (Validation)\n")
            for i, name in enumerate(CLASS_NAMES):
                try:
                    f.write(
                        f"    {name:<22} "
                        f"P={val_metrics.box.p[i]:.4f}  "
                        f"R={val_metrics.box.r[i]:.4f}  "
                        f"AP50={val_metrics.box.ap50[i]:.4f}\n"
                    )
                except Exception:
                    f.write(f"    {name:<22} metrics unavailable\n")
        except Exception as e:
            f.write(f"  Could not extract validation metrics: {e}\n")

        f.write("\n")

        # Test metrics
        f.write("TEST METRICS\n")
        try:
            f.write(f"  Precision    : {test_metrics.box.mp:.4f}\n")
            f.write(f"  Recall       : {test_metrics.box.mr:.4f}\n")
            f.write(f"  mAP@50       : {test_metrics.box.map50:.4f}\n")
            f.write(f"  mAP@50-95    : {test_metrics.box.map:.4f}\n\n")

            f.write("  Per-Class (Test)\n")
            for i, name in enumerate(CLASS_NAMES):
                try:
                    f.write(
                        f"    {name:<22} "
                        f"P={test_metrics.box.p[i]:.4f}  "
                        f"R={test_metrics.box.r[i]:.4f}  "
                        f"AP50={test_metrics.box.ap50[i]:.4f}\n"
                    )
                except Exception:
                    f.write(f"    {name:<22} metrics unavailable\n")
        except Exception as e:
            f.write(f"  Could not extract test metrics: {e}\n")

        f.write("\n" + "=" * 60 + "\n")

    print(f"\n[+] Training summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
