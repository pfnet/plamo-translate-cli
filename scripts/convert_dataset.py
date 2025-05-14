import argparse
import json
from pathlib import Path

from datasets import Dataset
from jinja2 import Template
from mlx_lm.tokenizer_utils import load_tokenizer
from mlx_lm.tuner import datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-path", type=str)
    parser.add_argument("--dataset-jsonl-path", type=str)
    parser.add_argument("--chat-template-path", type=str)
    parser.add_argument("--pack-length", type=int, default=640)
    args = parser.parse_args()

    tokenizer_path = args.tokenizer_path
    dataset_jsonl_path = args.dataset_jsonl_path
    chat_template_path = args.chat_template_path

    tokenizer = load_tokenizer(Path(tokenizer_path))

    with open(chat_template_path, "r") as f:
        chat_template = Template(f.read())

    with open(dataset_jsonl_path, "r") as f:
        lines = [json.loads(line) for line in f.readlines()]

    dataset = []
    prompts = []
    n_toks = []
    current_n_toks = 0
    for line in lines:
        for input_text, output_text in zip(line["input"]["content"], line["output"]["content"]):
            try:
                if "\n" in input_text:
                    input_text_str = input_text.split("\n")[1].strip()
                else:
                    input_text_str = input_text.strip()
            except Exception:
                print(input_text)
                import ipdb

                ipdb.set_trace()
            try:
                if "\n" in output_text:
                    output_text_str = output_text.split("\n")[1].strip()
                else:
                    output_text_str = output_text.strip()
            except Exception:
                print(output_text)
                import ipdb

                ipdb.set_trace()  # fmt: skip
            prompt = chat_template.render(
                messages=[
                    {"role": "user", "content": f"input lang={line['input']['lang']}\n{input_text_str}"},
                    {"role": "user", "content": f"output lang={line['output']['lang']}\n{output_text_str}"},
                ]
            )

            n_tok = len(tokenizer.encode(prompt.strip()))
            n_toks.append(n_tok)
            if current_n_toks + n_tok + 1 > args.pack_length:
                text = "<|plamo:bos|>".join(prompts) + "<|plamo:bos|>"
                n_pad = args.pack_length - len(tokenizer.encode(text))
                if n_pad > 0:
                    text += "<|plamo:pad|>" * n_pad
                dataset.append({"text": text})
                prompts = [prompt.strip()]
                current_n_toks = n_tok
            else:
                prompts.append(prompt.strip())
                current_n_toks += n_tok

    print(f"Max tokens in a batch: {max(n_toks)}")
    dataset = Dataset.from_list(dataset)
    dataset.save_to_disk("tmp/calibration_dataset")
