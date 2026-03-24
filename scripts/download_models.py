# ABOUTME: Downloads InsightFace model weights for face detection (RetinaFace) and recognition (ArcFace)
# ABOUTME: Models are cached locally so they don't need to be re-downloaded on each run

import argparse
import os


def download_models(model_dir: str) -> None:
    """Download InsightFace models using its built-in model zoo."""
    os.makedirs(model_dir, exist_ok=True)
    os.environ["INSIGHTFACE_HOME"] = model_dir

    from insightface.app import FaceAnalysis

    # Download buffalo_l model pack (includes RetinaFace + ArcFace)
    app = FaceAnalysis(
        name="buffalo_l",
        root=model_dir,
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))
    print(f"Models downloaded to {model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download InsightFace models")
    parser.add_argument("--model-dir", default="models", help="Directory to store models")
    args = parser.parse_args()
    download_models(args.model_dir)
