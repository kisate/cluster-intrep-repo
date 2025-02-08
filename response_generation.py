import os
import random

import yaml
# from Executor import Executor
from utils import *
from pathlib import Path
from tarski.io import PDDLReader
import argparse
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import json
from tqdm import tqdm

class ResponseGenerator:
    def __init__(self, config_file, engine, verbose, ignore_existing,temperature):
        self.engine = engine
        self.verbose = verbose
        self.ignore_existing = ignore_existing
        self.max_gpt_response_length = 500
        self.temperature = temperature
        self.data = self.read_config(config_file)
        if self.engine == 'bloom':
            self.model = self.get_bloom()
        elif 'finetuned' in self.engine:
            # print(self.engine)
            assert self.engine.split(':')[1] is not None
            model = ':'.join(self.engine.split(':')[1:])
            # print(model)
            self.engine='finetuned'
            self.model = {'model':model}
        else:
            self.model = None
    def read_config(self, config_file):
        with open(config_file, 'r') as file:
            return yaml.safe_load(file)
    def get_bloom(self):
        max_memory_mapping = {0: "0GB", 1: "43GB", 2: "43GB", 3: "43GB", 4: "43GB", 5: "43GB"}
        cache_dir = os.getenv('BLOOM_CACHE_DIR', '/data/karthik/LLM_models/bloom/')
        tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom")
        model = AutoModelForCausalLM.from_pretrained("bigscience/bloom", cache_dir=cache_dir,
                                                     local_files_only=False, load_in_8bit=True, device_map='auto',
                                                     max_memory=max_memory_mapping)
        return {'model': model, 'tokenizer': tokenizer}

    def get_responses(self, task_name, specified_instances = [], run_till_completion=False):
        output_dir = f"responses/{self.data['domain_name']}/{self.engine}/"
        os.makedirs(output_dir, exist_ok=True)
        output_json = output_dir+f"{task_name}.json"
        while True:
            if os.path.exists(output_json):
                with open(output_json, 'r') as file:
                    structured_output = json.load(file)
                prompt_dir = f"prompts/{self.data['domain_name']}/"
                assert os.path.exists(prompt_dir+f"{task_name}.json")
                with open(prompt_dir+f"{task_name}.json", 'r') as file:
                    prompts = json.load(file)
                prompts_instances = [inst['instance_id'] for inst in prompts['instances']]
                output_instances = [inst['instance_id'] for inst in structured_output['instances']]
                missing_instances = list(set(prompts_instances) - set(output_instances))
                structured_output['instances'] += [inst for inst in prompts['instances'] if inst['instance_id'] in missing_instances]
            else:
                prompt_dir = f"prompts/{self.data['domain_name']}/"
                assert os.path.exists(prompt_dir+f"{task_name}.json"), f"Prompt file {prompt_dir+f'{task_name}.json'} does not exist"
                with open(prompt_dir+f"{task_name}.json", 'r') as file:
                    structured_output = json.load(file)
                structured_output['engine'] = self.engine        
        
            failed_instances = []
            for instance in tqdm(structured_output["instances"],desc=f"Running {task_name}"):
                if "llm_raw_response" in instance:
                    if instance["llm_raw_response"] and not self.ignore_existing:
                        # if self.verbose:
                        print(f"Instance {instance['instance_id']} already completed")
                        continue
                if len(specified_instances) > 0:
                    if instance['instance_id'] not in specified_instances:
                        continue
                    else:
                        specified_instances.remove(instance['instance_id'])                   
                
                if self.verbose:
                    print(f"Sending query to LLM: Instance {instance['instance_id']}")
                query = instance["query"]
                stop_statement = "[STATEMENT]"
                if 'caesar' in self.data['domain_name']:
                    stop_statement = caesar_encode(stop_statement)
                llm_response = send_query(query, self.engine, self.max_gpt_response_length, model=self.model, stop=stop_statement, temperature=self.temperature)
                if not llm_response:
                    failed_instances.append(instance['instance_id'])
                    print(f"Failed instance: {instance['instance_id']}")
                    continue
                if self.verbose:
                    print(f"LLM response: {llm_response}")
                instance["llm_raw_response"] = llm_response
                with open(output_json, 'w') as file:
                    json.dump(structured_output, file, indent=4)
            
            if run_till_completion:
                if len(failed_instances) == 0:
                    break
                else:
                    print(f"Retrying failed instances: {failed_instances}")
                    time.sleep(5)
            else:
                break
    
    def get_multiple_responses(self, task_name, params, specified_instances = [], run_till_completion=False):
        output_dir = f"responses/{self.data['domain_name']}/{self.engine}/"
        os.makedirs(output_dir, exist_ok=True)
        prompts_task_name =task_name
        task_name = f"{task_name}_n{params['n']}_temp{'_'.join(str(params['temperature']).split('.'))}"
        output_json = output_dir+f"{task_name}.json"
        while True:
            if os.path.exists(output_json):
                with open(output_json, 'r') as file:
                    structured_output = json.load(file)
                prompt_dir = f"prompts/{self.data['domain_name']}/"
                assert os.path.exists(prompt_dir+f"{prompts_task_name}.json")
                with open(prompt_dir+f"{prompts_task_name}.json", 'r') as file:
                    prompts = json.load(file)
                prompts_instances = [inst['instance_id'] for inst in prompts['instances']]
                output_instances = [inst['instance_id'] for inst in structured_output['instances']]
                missing_instances = list(set(prompts_instances) - set(output_instances))
                structured_output['instances'] += [inst for inst in prompts['instances'] if inst['instance_id'] in missing_instances]
            else:
                prompt_dir = f"prompts/{self.data['domain_name']}/"
                assert os.path.exists(prompt_dir+f"{prompts_task_name}.json")
                with open(prompt_dir+f"{prompts_task_name}.json", 'r') as file:
                    structured_output = json.load(file)
                structured_output['engine'] = self.engine        
        
            failed_instances = []
            for instance in tqdm(structured_output["instances"]):
                if "llm_raw_response" in instance:
                    if instance["llm_raw_response"] and not self.ignore_existing:
                        # if self.verbose:
                        print(f"Instance {instance['instance_id']} already completed")
                        continue
                if len(specified_instances) > 0:
                    if instance['instance_id'] not in specified_instances:
                        continue
                    else:
                        specified_instances.remove(instance['instance_id'])                   
                
                if self.verbose:
                    print(f"Sending query to LLM: Instance {instance['instance_id']}")
                query = instance["query"]
                stop_statement = "[STATEMENT]"
                if 'caesar' in self.data['domain_name']:
                    stop_statement = caesar_encode(stop_statement)
                llm_response = send_query_multiple(query, self.engine, self.max_gpt_response_length, params, model=self.model, stop=stop_statement)
                if not llm_response:
                    failed_instances.append(instance['instance_id'])
                    print(f"Failed instance: {instance['instance_id']}")
                    continue
                if self.verbose:
                    print(f"LLM response: {llm_response}")
                instance["llm_raw_response"] = llm_response
                with open(output_json, 'w') as file:
                    json.dump(structured_output, file, indent=4)
            
            if run_till_completion:
                if len(failed_instances) == 0:
                    break
                else:
                    print(f"Retrying failed instances: {failed_instances}")
                    time.sleep(5)
            else:
                break

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', '-t',type=str, required=True, help='Task to run \
    \n cot = Plan Generation COT\
    ')
    parser.add_argument('--engine', '-e', type=str, required=True, help='Engine to use \
                        \n gpt-4_chat = GPT-4 \
                        \n bloom = Bloom \
                        \n gpt-3.5-turbo_chat = GPT-3.5 Turbo \
                        \n davinci = GPT-3 Davinci \
                        \n curie = GPT-3 Curie \
                        \n babbage = GPT-3 Babbage \
                        \n ada = GPT-3 Ada \
                        ')
    parser.add_argument('--verbose', '-v',type=str, default="False", help='Verbose')
    #config
    parser.add_argument('--config', '-c', type=str, required=True, help='Config file name (no need to add .yaml)')
    parser.add_argument('--specific_instances', '-si', nargs='+', type=int, default=[], help='List of instances to run')
    parser.add_argument('--run_till_completion', '-rc', action='store_true', help='Run till completion')
    parser.add_argument('--ignore_existing', '-ie', action='store_true', help='Ignore existing output')
    parser.add_argument('--temperature', '-temp', type=float, default=0, help='Random seed')
    parser.add_argument('--cot_type', '-ct', type=str, default="none", help='COT type \
    \n upb = Universal Plan Implicit Breakdown\
    ')
   
    parser.add_argument('--additional_arguments', '-aa', type=str, default = '',help='Additional info\
                        \n sg = stack generalization \
                        \n pm = pattern matching \
                        \n brb = backroll block\
                        \n brsb = backroll stack block \
                        ')
    parser.add_argument('--override_example', '-oe', type=str, default="none", help='Override example\
                        \n st = State Tracking \
                        \n ds = domain specific \
                        \n none = None')
    parser.add_argument('--n', '-n', type=int, default=1, help='Number of responses')
    args = parser.parse_args()
    task = args.task
    engine = args.engine
    config = args.config
    verbose = eval(args.verbose)
    specified_instances = args.specific_instances
    ignore_existing = args.ignore_existing
    run_till_completion = args.run_till_completion
    temperature = args.temperature
    order = 'po'
    add_args = args.additional_arguments
    override_example = args.override_example
    n = args.n
    if override_example not in ['st', 'ds', 'lex','none']:
        raise ValueError('Not in override example')
    if add_args not in ['sg', 'pm','brb','']:
        raise ValueError('Not in add_arg')
    # print(task, config, verbose, specified_instances, random_example)
    config_file = f'./configs/{config}.yaml'
    assert os.path.exists(config_file), f"Config file {config_file} does not exist"
    if args.cot_type not in ["upb", "none","zshot"]:
        raise ValueError("Invalid COT type")
    else:
        # cot_dict = {
        #     "upi": "universal_plan_implicit",
        #     "upis": "universal_plan_implicit_serialize",
        #     "upb": "universal_plan_implicit_breakdown",
        #     "upeb": "universal_plan_explicit_breakdown",
        #     "sg": "subgoals",
        #     "lm": "landmarks",
        #     "none": "none"
        # }
        # cot_type = cot_dict[args.cot_type]
        cot_type = args.cot_type

    task_dict = {
        'cot': 'plan_generation_cot',
        'standard': 'plan_generation',
        'standard_zeroshot': 'plan_generation_zshot',
    } 
    
     
    if (task=="standard" or task=="standard_zeroshot") and cot_type!="none":
        raise ValueError("COT type not supported for standard plan generation")
    if task not in task_dict:
        raise ValueError("Invalid task")
    else:
        task_name = task_dict[task]+f"_{add_args}" if add_args else task_dict[task]
        task_name += f"_{cot_type}" if cot_type != "none" else ""
        task_name += f"_{order}" if not add_args else ""
        task_name += f"_{override_example}" if override_example != "none" else ""
    print(f"Task: {task}, Config: {config}, Verbose: {verbose}, Ignore existing: {ignore_existing}")
    response_generator = ResponseGenerator(config_file, engine, verbose, ignore_existing, temperature)
    if n == 1:
        response_generator.get_responses(task_name, specified_instances, run_till_completion)
    elif n>1:
        params = {
            'temperature': temperature,
            'n': n
        }
        response_generator.get_multiple_responses(task_name, params, specified_instances, run_till_completion)