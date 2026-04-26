"""
Submit the Cross-Session Continuity GRPO training job to HF infrastructure.

Cost: L4 GPU = ~$0.80/hr. Training takes ~1 hour. Total cost: < $1.00 from your $30 credits.

Usage:
    set HF_TOKEN=hf_yourtoken
    python submit_job.py
"""
import os, sys
try:
    from huggingface_hub import HfApi
except ImportError:
    print("pip install huggingface_hub to run this script")
    sys.exit(1)

TOKEN = os.environ.get("HF_TOKEN") or input("Enter your HF token (hf_...): ").strip()

# The Space we are deploying on
SPACE_ID = "Aswini-Kumar/cross-session-continuity-env"

api = HfApi(token=TOKEN)

print("Submitting HF training job...")
print(f"  Target Space : {SPACE_ID}")
print(f"  Hardware     : nvidia-l4 (~$0.80/hr)")

try:
    job = api.run_job(
        image="pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime",
        command=[
            "bash", "-c",
            "apt-get update && apt-get install -y git && "
            "git clone https://github.com/CelestialWorthyOfHeavenAndEarth/cross-session-continuity-env.git && "
            "cd cross-session-continuity-env && "
            "pip install -r requirements.txt && "
            "pip install -q 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git' "
            "trl>=0.15.0 datasets>=2.0.0 transformers>=4.40.0 accelerate>=0.30.0 matplotlib && "
            "python training/hf_job_train.py"
        ],
        env={
            "HF_TOKEN": TOKEN,
        },
        flavor="l4x1",  # Using L4 since 7B model training needs good VRAM
    )

    print(f"\n✅ Job submitted successfully!")
    print(f"   Job ID  : {job.id}")
    print(f"   Status  : {job.status}")
    print(f"   Monitor : https://huggingface.co/spaces/{SPACE_ID}/jobs")
    print(f"\nWhen DONE (~1 hr), the model will automatically be pushed to the hub!")
except Exception as e:
    print(f"\n❌ Error submitting job: {e}")
    print("\nAlternative: Use Hugging Face Spaces 'JupyterLab' template.")
    print("1. Go to huggingface.co/spaces and create a new Docker space.")
    print("2. Choose 'JupyterLab' template.")
    print("3. Set Hardware to L4 or A10G.")
    print("4. Clone the repo and run `train_grpo.ipynb` directly.")
