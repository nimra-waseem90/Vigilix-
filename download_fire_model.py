from huggingface_hub import hf_hub_download
import shutil
import os

model_path = hf_hub_download(
    repo_id="rabahdev/fire-smoke-yolov8n",
    filename="best.pt"
)

destination = "model/detection/fire_smoke_model.pt"

os.makedirs("model/detection", exist_ok=True)

shutil.copy(model_path, destination)

print("Fire/Smoke model downloaded successfully!")
print("Saved to:", destination)