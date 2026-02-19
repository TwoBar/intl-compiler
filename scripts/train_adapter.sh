#!/bin/bash
# INTL Adapter Training Script — Vast.ai RTX 4090
# Usage: bash scripts/train_adapter.sh <adapter_name>
#
# Prerequisites:
#   - VASTAI_API_KEY set
#   - HF_TOKEN set
#   - Training data at data/<adapter>/train.jsonl and data/<adapter>/val.jsonl

set -euo pipefail

ADAPTER="${1:?Usage: train_adapter.sh <adapter_name>}"
DATA_DIR="data/${ADAPTER}"
HF_REPO="bballin22/intl-adapters"
TRAINING_REPO="bballin22/intl-training-pairs"

echo "═══════════════════════════════════════"
echo "INTL Adapter Training: ${ADAPTER}"
echo "═══════════════════════════════════════"

# Validate training data exists
if [[ ! -f "${DATA_DIR}/train.jsonl" ]]; then
    echo "ERROR: ${DATA_DIR}/train.jsonl not found. Run datagen first."
    exit 1
fi

TRAIN_COUNT=$(wc -l < "${DATA_DIR}/train.jsonl")
VAL_COUNT=$(wc -l < "${DATA_DIR}/val.jsonl")
echo "Training pairs: ${TRAIN_COUNT}"
echo "Validation pairs: ${VAL_COUNT}"

# Estimate cost
echo ""
echo "Estimated cost: ~$0.15 (25 min @ $0.32/hr RTX 4090)"
echo ""

# Search for cheapest RTX 4090
echo "Searching for RTX 4090 instances..."
INSTANCE_ID=""

# Find cheapest 4090
OFFER=$(vastai search offers 'gpu_name=RTX_4090 num_gpus=1 inet_down>200 disk_space>40' \
    -o 'dph_total' --raw 2>/dev/null | python3 -c "
import json, sys
offers = json.load(sys.stdin)
if offers:
    best = offers[0]
    print(f'{best[\"id\"]}|{best[\"dph_total\"]:.4f}')
else:
    print('NONE')
")

if [[ "$OFFER" == "NONE" ]]; then
    echo "ERROR: No RTX 4090 available on Vast.ai"
    exit 1
fi

OFFER_ID=$(echo "$OFFER" | cut -d'|' -f1)
OFFER_PRICE=$(echo "$OFFER" | cut -d'|' -f2)
echo "Best offer: #${OFFER_ID} at \$${OFFER_PRICE}/hr"

# Create instance
echo "Creating instance..."
INSTANCE_ID=$(vastai create instance ${OFFER_ID} \
    --image "ghcr.io/unslothai/unsloth:latest" \
    --disk 40 \
    --raw | python3 -c "import json,sys; print(json.load(sys.stdin)['new_contract'])")

echo "Instance created: ${INSTANCE_ID}"
echo "Waiting for instance to be ready..."

# Wait for instance
for i in $(seq 1 60); do
    STATUS=$(vastai show instance ${INSTANCE_ID} --raw | python3 -c "import json,sys; print(json.load(sys.stdin)['actual_status'])" 2>/dev/null || echo "pending")
    if [[ "$STATUS" == "running" ]]; then
        echo "Instance is running!"
        break
    fi
    echo "  Status: ${STATUS} (attempt ${i}/60)"
    sleep 10
done

# Upload training data and run training
echo "Uploading training data and starting training..."
vastai copy "${DATA_DIR}/train.jsonl" ${INSTANCE_ID}:/workspace/train.jsonl
vastai copy "${DATA_DIR}/val.jsonl" ${INSTANCE_ID}:/workspace/val.jsonl

# Execute training script on remote instance
vastai execute ${INSTANCE_ID} "bash -c '
pip install -q huggingface_hub
export HF_TOKEN=\"${HF_TOKEN}\"

python3 -c \"
from unsloth import FastLanguageModel
import json, os

# Load base model with 4-bit quantization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=\"Qwen/Qwen2.5-Coder-3B-Instruct\",
    max_seq_length=2048,
    load_in_4bit=True,
    dtype=None,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    target_modules=[\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\", \"gate_proj\", \"up_proj\", \"down_proj\"],
)

# Load training data
from datasets import Dataset
train_data = [json.loads(l) for l in open(\"/workspace/train.jsonl\")]
val_data = [json.loads(l) for l in open(\"/workspace/val.jsonl\")]

def format_example(ex):
    return {\"text\": ex[\"system\"] + \"\\n\" + ex[\"prompt\"] + \"\\n\" + ex[\"completion\"] + tokenizer.eos_token}

train_ds = Dataset.from_list(train_data).map(format_example)
val_ds = Dataset.from_list(val_data).map(format_example)

# Train
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    dataset_text_field=\"text\",
    max_seq_length=2048,
    args=TrainingArguments(
        output_dir=\"/workspace/output\",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        lr_scheduler_type=\"cosine\",
        warmup_ratio=0.05,
        logging_steps=10,
        eval_strategy=\"epoch\",
        save_strategy=\"epoch\",
        bf16=True,
    ),
)

trainer.train()

# Save and push LoRA adapter
model.save_pretrained(\"/workspace/output/lora\")
tokenizer.save_pretrained(\"/workspace/output/lora\")

from huggingface_hub import HfApi
api = HfApi(token=os.environ[\"HF_TOKEN\"])
api.upload_folder(
    folder_path=\"/workspace/output/lora\",
    repo_id=\"${HF_REPO}\",
    path_in_repo=\"${ADAPTER}\",
    repo_type=\"model\",
)
print(\"TRAINING_COMPLETE\")
\"
'"

echo "Training submitted. Monitoring..."

# Wait for completion (check logs)
for i in $(seq 1 120); do
    LOGS=$(vastai logs ${INSTANCE_ID} 2>/dev/null | tail -5)
    if echo "$LOGS" | grep -q "TRAINING_COMPLETE"; then
        echo "Training complete!"
        break
    fi
    echo "  Training in progress... (check ${i}/120)"
    sleep 15
done

# CRITICAL: Always destroy instance
echo "Destroying instance ${INSTANCE_ID}..."
vastai destroy instance ${INSTANCE_ID}
echo "Instance destroyed."

echo ""
echo "═══════════════════════════════════════"
echo "Adapter ${ADAPTER} trained and pushed to:"
echo "  https://huggingface.co/${HF_REPO}/tree/main/${ADAPTER}"
echo "═══════════════════════════════════════"
