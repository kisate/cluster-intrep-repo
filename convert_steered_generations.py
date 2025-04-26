import json

from pathlib import Path

def main_gl(mystery_n, copy_n=None):
    from_intermediate = False
    domain = f"blocksworld_mystery_{mystery_n}"

    if not from_intermediate:
        file_path = f"results/mystery_replaced_3000_4500_8_30_47_r/steered_results_mystery_{mystery_n}.json"
        dataset = json.load(open(file_path, "r"))
    else:
        path = Path("intermediate_results/intermediate_results_9_full")
        
        dataset = []
        
        for file in path.glob("*.json"):
            with open(file, "r") as f:
                data = json.load(f)
                dataset.extend(data)
                
    print(
        len(dataset),
    )

    # print(
    #     [[x["idx"], len(x["original_input"])] for x in dataset]
    # )
                
    def main(steered_generation):
        instances = {}

        all_ids = set()

        cf = 0

        for x in dataset:
            if copy_n is not None and x.get("copy", -1) != copy_n:
                continue
            iid = f"4_{str(x['idx'] + 1)}"
            if iid in instances:
                continue

            all_ids.add(iid)
            
            if steered_generation:
                field_name = "steered_generation"
            else:
                field_name = "original_input"

            raw_llm_answer = x[field_name]
            
            if "</think>" not in raw_llm_answer:
                raw_llm_answer = "Still thinking..."
                cf += 1
            else:
                raw_llm_answer = raw_llm_answer.split("</think>")[1].strip()
            

            instances[iid] = {
                "instance_id": iid,
                "Number of blocks": 4,
                "llm_raw_response": raw_llm_answer,
                "full_response": x[field_name],
                "dataset_idx": x["idx"],
            }

        print(cf / len(dataset))

        formatted_json = {}

        formatted_json["task"] = "plan_generation_po"
        formatted_json["instances"] = list(instances.values())
        formatted_json["prompt_type"] = "fewshot"
        formatted_json["domain"] = domain

        import json
        from pathlib import Path
        
        suffix = ""
        if copy_n is not None:
            suffix = f"_{copy_n}"
        
        if steered_generation:
            final_dir = Path(f"cot-planning/responses/{formatted_json['domain']}/qwq-32b-replaced-full-3000-4500-8-30-47-r{suffix}/")
        else:
            final_dir = Path(f"cot-planning/responses/{formatted_json['domain']}/qwq-32b-original-full-3000-4500-8-30-47-r{suffix}/")
            

        final_dir.mkdir(parents=True, exist_ok=True)

        with open(final_dir / f"{formatted_json['task']}.json", "w") as f:
            json.dump(formatted_json, f, indent=2)


    main(True)
    main(False)


for i in range(1, 16):
    try:
        ci = None
        # for ci in range(3):
        main_gl(i, ci)
    except Exception as e:
        print(f"Error processing mystery_{i}: {e}")
        continue