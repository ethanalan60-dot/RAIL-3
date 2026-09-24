"""Identity constants for the committed M06-PILOT-B protocol."""

PROTOCOL_SHA256 = "78ddfe0bfac1a9eeaba0858a49abeaa6cd21b64b0716521d65255630a89c8b4d"
CONFIG_RELATIVE_PATH = "configs/experiments/voc_m06_pilot_b_training_v1.json"
FOLD_SEMANTIC_SHA256 = "b2be128932fd4a0f2a9e151e0a5a123d60f3148ef8f8ef63ba701b72159c9b6a"
TARGET_MANIFEST_SHA256 = "40163970fd2ed37cae969bbe63eaaa3664ac27923862c07b98e18325ac6bc7e9"

MODEL_IDS = (
    "Q1_LINEAR_QG", "Q2_QG_MLP", "P1_LINEAR_PAL", "P2_PAL_MLP",
    "J1_Q_PAL_CONSISTENT",
)
ACTION_CODES = ("A0", "A1", "A2", "A3", "A4", "A5", "A6")
FOLD_IDS = ("fold_0", "fold_1", "fold_2", "fold_3", "fold_4")
