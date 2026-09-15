from pathlib import Path
import torch
from ultralytics import YOLO


def main():

    # Project root directory
    base_dir = Path(__file__).resolve().parent.parent.parent

    # Dataset path
    data_yaml = base_dir / "datasets" / "night_traffic" / "data.yaml"

    # Results directory
    project_dir = base_dir / "results" / "detection"

    print("=" * 60)
    print("       YOLO26 NIGHT TRAFFIC DETECTION TRAINING")
    print("=" * 60)

    print(f"[+] Repository Root   : {base_dir}")
    print(f"[+] Dataset Config    : {data_yaml}")
    print(f"[+] Output Directory  : {project_dir / 'night_traffic'}")


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


    # Load YOLO26 pretrained model
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
        name="night_traffic",

        # Prevent overwriting previous experiments
        exist_ok=False
    )


    print("\n" + "=" * 60)
    print("       TRAINING COMPLETE")
    print("=" * 60)

    output_dir = project_dir / "night_traffic"

    print(f"[+] Best Model : {output_dir / 'weights' / 'best.pt'}")
    print(f"[+] Last Model : {output_dir / 'weights' / 'last.pt'}")
    print(f"[+] Results    : {output_dir}")


if __name__ == "__main__":
    main()
