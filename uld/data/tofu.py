import json
import os
import datasets

from .conv_util import create_template
from .datamodule import TrainDataModule


class Deepseek_DataModule(TrainDataModule):
    """DataModule for deepseek code-completion unlearning.

    Loads a single JSON file (D_forget.json) and splits it:
      - forget: probing input + y_neg (deprecated API completions to unlearn)
      - retain: probing input + y_pos (correct API completions to preserve)
    """

    def __init__(
        self,
        split,
        tokenizer,
        conv_template_config,
        max_len=512,
        batch_size=8,
        with_retain=True,
        retain_num=400,
        retain_match_forget=False,
        **kwargs,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.batch_size = batch_size
        self.dpo_mode = False
        self.conv_template = create_template(conv_template_config, tokenizer=tokenizer)

        # Locate the data file
        data_root = kwargs.get("name", "../Data-Collection/deepseek")
        data_path = os.path.join(data_root, f"{split}.json")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Deepseek data file not found: {data_path}")

        print(f"[Deepseek] loading {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # Build forget (probing input + y_neg) and retain (probing input + y_pos)
        forget_list = [{"question": r["probing input"], "answer": r["y_neg"]} for r in raw if "y_neg" in r]
        retain_list = [{"question": r["probing input"], "answer": r["y_pos"]} for r in raw if "y_pos" in r]

        base_forget_data = datasets.Dataset.from_list(forget_list)
        base_retain_data = datasets.Dataset.from_list(retain_list)

        self.forget_length = len(base_forget_data)
        self.retain_length = 0

        if with_retain and len(retain_list) > 0:
            if retain_match_forget:
                retain_num = min(retain_num, len(base_forget_data))
            retain_num = min(retain_num, len(base_retain_data))
            base_retain_data = base_retain_data.select(
                range(len(base_retain_data) - retain_num, len(base_retain_data))
            )
            self.retain_length = len(base_retain_data)

        self.forget_data = datasets.concatenate_datasets([base_forget_data, base_retain_data])

        # Dummy eval sets (deepseek does not use ToFU-style eval)
        dummy = datasets.Dataset.from_list([{"question": "", "answer": ""}])
        self.eval_sets = {"forget": dummy, "retain": dummy}

        print(f"[Deepseek] Train: forget={self.forget_length}, retain={self.retain_length}")
