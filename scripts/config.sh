# ============================================================
# CARF-Net configuration. Edit the paths here only.
# ============================================================

# Raw frame roots
export RAW_casme2="/data/CASME2_RAW_selected"   # sub01/EP03_02/img131.jpg
export RAW_samm="/data/SAMM"                    # 006/006_1_2/006_05562.jpg
export RAW_casme3="/data/CASME3"                # spNo.1_a_355/355.jpg

# Official annotation files (obtain them from the dataset owners)
export LABEL_casme2="/data/CASME2-coding-20140508.xlsx"
export LABEL_samm="/data/SAMM_Micro_FACS_Codes_v2.xlsx"
export LABEL_casme3="/data/cas(me)3_part_A_ME_label_JpgIndex_v2.xlsx"

# Where the processed clips are written (one subfolder per dataset)
export PROCESSED_ROOT="/data/carf_processed"

# Number of classes (3 = positive, negative, surprise)
export CLASSES="${CLASSES:-3}"

# Training
export DEVICE="${DEVICE:-cuda:0}"
export SEEDS="${SEEDS:-0,1,2}"
export EPOCHS="${EPOCHS:-60}"

# Checkpoint-selection protocols to compute; see carfnet/train.py
export PROTOCOLS="${PROTOCOLS:-test_peek}"

# Per-fold checkpoints for Grad-CAM (about 100 MB per file)
#   SAVE_CKPT  : any protocol listed in PROTOCOLS, a comma list, or none
#   CKPT_SEEDS : first | all | a list such as 0,2
export SAVE_CKPT="${SAVE_CKPT:-test_peek}"
export CKPT_SEEDS="${CKPT_SEEDS:-first}"
