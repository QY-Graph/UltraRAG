import json
import yaml
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from vllm import LLM, SamplingParams
import argparse
from pathlib import Path
import sys
home_path = Path().resolve()
sys.path.append(home_path.as_posix())

from ultrarag.modules.llm import VllmServer

class DPOGenerator:
    def __init__(self, input_path, output_path, model_name_or_path, config_path):
        self.input_path = input_path
        self.output_path = output_path
        self.model_name_or_path = model_name_or_path
        self.config_path = config_path

        self.config = self.load_yaml_config(config_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.vllm_params = self.config["VllmServer_params"]
        self.llm_service = VllmServer(base_url=self.model_name_or_path, **self.vllm_params)

        self.batch_size = self.config.get('batch_size', 4)
        self.is_llama_style = self.config.get('is_llama_style', True)
        self.Augment_template = self.config.get('Augment_template', 'Background:\n{}\n\nQuestion: {}\nAnswer:')
        self.QA_template = self.config.get('QA_template', 'Question: {}\nAnswer:')

    def load_yaml_config(self, config_path):
        """Load YAML configuration file."""
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config

    def read_jsonl(self):
        """Read the input JSONL file."""
        data = []
        with open(self.input_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                item = json.loads(line.strip())
                item['id'] = idx
                data.append(item)
        return data
    
    def create_dataset(self, input_data):
        """Create the dataset using the input data and tokenizer."""
        return DPODataset(input_data, self.tokenizer, self.config)

    def generate(self, dataloader, temperature_list, sampling_params_dict):
        """Run the model and generate results."""
        input_data = self.read_jsonl()
        all_save_list = {item['id']: {'id': item['id'], 'context': []} for item in input_data}

        aug_outputs = []
        raw_outputs = []

        for temp in tqdm(temperature_list):
            params_dict = {
                "n": sampling_params_dict.get('n', 5),
                "best_of": sampling_params_dict.get('best_of', 5),
                "presence_penalty": sampling_params_dict.get('presence_penalty', 1.0),
                "frequency_penalty": sampling_params_dict.get('frequency_penalty', 0.0),
                "temperature": temp,
                "top_p": sampling_params_dict.get('top_p', 0.8),
                "top_k": sampling_params_dict.get('top_k', -1),
                "stop": sampling_params_dict.get('stop', None),
                "stop_token_ids": sampling_params_dict.get('stop_token_ids', None),
                "ignore_eos": sampling_params_dict.get('ignore_eos', False),
                "max_tokens": sampling_params_dict.get('max_tokens', 100),
                "logprobs": sampling_params_dict.get('logprobs', None),
                "prompt_logprobs": sampling_params_dict.get('prompt_logprobs', None),
                "skip_special_tokens": sampling_params_dict.get('skip_special_tokens', True),
            }

            sampling_params = SamplingParams(**params_dict)

            for batch in tqdm(dataloader):
                aug_inputs = batch['augment_input']
                raw_inputs = batch['raw_input']
                ids = batch['ids']

                aug_outputs = self.llm_service._generator.generate(aug_inputs, sampling_params)
                raw_outputs = self.llm_service._generator.generate(raw_inputs, sampling_params)

                for idx, raw_output in enumerate(raw_outputs):
                    generated_raw_text = raw_output.outputs[0].text
                    all_save_list[ids[idx]]['context'].append({
                        'text': generated_raw_text, 
                        'temperature': temp, 
                        'type': 'raw'
                    })
                
                for idx, aug_output in enumerate(aug_outputs):
                    generated_aug_text = aug_output.outputs[0].text
                    all_save_list[ids[idx]]['context'].append({
                        'text': generated_aug_text, 
                        'temperature': temp, 
                        'type': 'aug'
                    })

        return all_save_list

    def update_results(self, input_data, all_save_list):
        """Update the generated results in the input data."""
        updated_data = []
        for item in input_data:
            item_id = item.get('id')
            if item_id in all_save_list:
                item['context'] = all_save_list[item_id]['context']
            updated_data.append(item)
        return updated_data

    def save_results(self, updated_data):
        """Save the updated results to the output file."""
        with open(self.output_path, 'w', encoding='utf-8') as f:
            for item in updated_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    def run(self):
        """Run the data generation process."""
        input_data = self.read_jsonl()
        dataset = self.create_dataset(input_data)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, collate_fn=dataset.collator)

        sampling_params_dict = self.config.get('dpo_sampling_params', {})
        sampling_params_dict['temperature'] = sampling_params_dict.get('temperature', [0.5, 0.6, 0.7, 0.8, 0.9])
        temperature_list = sampling_params_dict['temperature']

        all_save_list = self.generate(dataloader, temperature_list, sampling_params_dict)
        updated_data = self.update_results(input_data, all_save_list)
        self.save_results(updated_data)
        print(f"DPO data completed, results have been saved in real-time to {self.output_path}")

class DPODataset(Dataset):
    def __init__(self, data_list, tokenizer, args):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.args = args
        self.top_k = self.args.get('top_k', 5)
        self.sep_token = args.get('passage_separator', "\n")
        self.max_passage_length = self.args.get('max_passage_length', 2000)
        self.model_type = self.args.get('model_type', "minicpm3")
        self.use_template = self.args.get('use_template', True)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        item = self.data_list[index]

        if 'raw_input' not in item:
            item['raw_input'] = []
        if 'augment_input' not in item:
            item['augment_input'] = []

        query = item['query']
        retrieve_result = item['retrieval_result']
        retrieve_result = retrieve_result[:self.top_k]

        passage = self.sep_token.join(retrieve_result)
        new_aug_psg = self.truncated_passage(passage, self.tokenizer, self.max_passage_length)

        if self.model_type == "minicpm2":
            aug_input = f"<User>{self.args['Augment_template'].format(new_aug_psg, query)}<AI>"
            raw_input = f"<User>{self.args['QA_template'].format(query)}<AI>"
            
        elif self.model_type == "minicpm3":
            aug_input = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": self.args['Augment_template'].format(new_aug_psg, query)},
            ]
            aug_input = self.tokenizer.apply_chat_template(aug_input, add_generation_prompt=True, tokenize=False)
            raw_input = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": self.args['QA_template'].format(query)},
                ]
            raw_input = self.tokenizer.apply_chat_template(raw_input, add_generation_prompt=True, tokenize=False)

        else:
            if self.use_template:
                aug_input = [{"role": "user", "content": self.args['Augment_template'].format(new_aug_psg, query)}]
                aug_input = self.tokenizer.apply_chat_template(aug_input, add_generation_prompt=True, tokenize=False)
                raw_input = [{"role": "user", "content": self.args['QA_template'].format(query)}]
                raw_input = self.tokenizer.apply_chat_template(raw_input, add_generation_prompt=True, tokenize=False)
                
            else:
                aug_input = self.args['Augment_template'].format(new_aug_psg, query)
                raw_input = self.args['QA_template'].format(query)

        item['augment_input'] = aug_input
        item['raw_input'] = raw_input

        return item
    
    def truncated_passage(self, passage, tokenizer, truncate_size):
        encoded_passage = tokenizer.encode(passage, add_special_tokens=False)
        truncated_encoded_passage = encoded_passage[:truncate_size]
        decoded_passage = tokenizer.decode(truncated_encoded_passage)
        return decoded_passage

    def collator(self, batch):     
        query = [f['query'] for f in batch]
        id = [f['id'] for f in batch]
        ground_truth = [f['ground_truth'] for f in batch]
        raw_input = [f['raw_input'] for f in batch]
        augment_input = [f['augment_input'] for f in batch]
        keypoints = [f['keypoints'] for f in batch]

        return {
            'ids': id,
            'query': query,
            'ground_truth': ground_truth,
            'raw_input': raw_input,
            'augment_input': augment_input,
            'keypoints': keypoints,
        }

def main():
    parser = argparse.ArgumentParser(description="Generate DPO data candidates")
    parser.add_argument('--input_path', type=str, required=True, help="Path to save the retrieval results (JSONL format).")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the generate results (JSONL format).")
    parser.add_argument('--model_name_or_path', type=str, required=True, help="Path to the model to be trained.")
    parser.add_argument('--config_path', type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    generator = DPOGenerator(args.input_path, args.output_path, args.model_name_or_path, args.config_path)
    generator.run()

if __name__ == '__main__':
    main()