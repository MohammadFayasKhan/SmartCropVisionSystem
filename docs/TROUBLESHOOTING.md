# SmartCropVision Operational Troubleshooting Playbook

## 1. Quick Diagnostic Triage Matrix

| Issue Symptom | Root Cause | Immediate Resolution Command |
| :--- | :--- | :--- |
| Backend reports HTTP 503 on `/health/ready` | One or more model weight files are missing from `cv/models/` | Run `python3 -m pytest tests/test_inference_pipeline.py` to identify missing checkpoints |
| Checksum mismatch warning during startup | Model file was altered, truncated, or replaced | Compare hash with `RELEASE_MANIFEST.json` and restore verified artifact |
| Image rejected with `IMAGE_VALIDATION_FAILED` | Image exceeds 16 million pixels or is not a valid JPEG/PNG | Resize leaf photograph to standard resolution before upload |
| Compute device shows `CPU` instead of `MPS` or `CUDA` | PyTorch was installed without hardware acceleration kernels | Check hardware availability: `python3 -c "import torch; print(torch.backends.mps.is_available())"` |
| Frontend displays `Connection Error` | Backend server process is not running or listening on port 8000 | Verify backend service: `curl -I http://localhost:8000/health` |

---

## 2. Detailed Failure Scenarios & Remedies

### 2.1 Model Checkpoint Missing or Corrupted
* **Symptom**: Server logs state `Missing required model checkpoint` and readiness probes return `503 Service Unavailable`.
* **Verification**:
  ```bash
  python3 -c "
  import hashlib
  p = 'cv/models/efficientnetv2_s_best.pt'
  print(hashlib.sha256(open(p, 'rb').read()).hexdigest())
  "
  ```
* **Remedy**: Compare output with the expected hash in `RELEASE_MANIFEST.json` (`07e84d39481f0757...`). Restore the verified file if checksums differ.

### 2.2 Taxonomy or Preprocessing Configuration Missing
* **Symptom**: Server raises `FileNotFoundError` for `taxonomy_38.json` or `preprocessing_config.json`.
* **Remedy**: Confirm that `cv/models/taxonomy_38.json` exists. The configuration defines the canonical 38 class mapping and ImageNet normalization statistics.

### 2.3 Out of Distribution Rejection
* **Symptom**: The backend returns a diagnosis with `ood_status: OUT_OF_DISTRIBUTION` and low confidence.
* **Explanation**: The uploaded photograph failed botanical quality screening (for example, non foliar background, extreme shadows, or severe motion blur).
* **Remedy**: Instruct the grower to hold the camera steady 15 centimeters from the leaf blade under natural indirect sunlight and retake the photograph.

### 2.4 GPU Memory Race Collision
* **Symptom**: Server logs show CUDA or MPS tensor allocation errors during concurrent requests.
* **Remedy**: SmartCropVision enforces inference thread locks inside `backend/app/services/inference_service.py`. Ensure that `InferenceEngine.get_instance()` is utilized as a singleton rather than instantiating multiple engines in parallel threads.
