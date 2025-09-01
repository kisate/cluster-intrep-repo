import json

from pathlib import Path

layer = 44
scale = "1"
start_t = 1000
end_t = 2500
steering = True

def main_gl(mystery_n, copy_n=None, suffix=""):
    from_intermediate = False
    domain = f"blocksworld_mystery_{mystery_n}"

    if suffix != "":
        in_suffix = f"_{suffix}"
        out_suffix = f"-{suffix}"
    else:
        in_suffix = ""
        out_suffix = ""

    if not from_intermediate:
        if steering:
            if scale == "0":
                file_path = f"results/mystery_steered_end_t_{start_t}_{end_t}_s_{scale}{in_suffix}/steered_results_mystery_{mystery_n}.json"
                file_path = f"results/mystery_steered_end_t_{start_t}_{end_t}_s_{scale}{in_suffix}/steered_results_mystery_{mystery_n}.json"
                file_path = f"results/mystery_steered_300_t_{start_t}_{end_t}_s_{scale}{in_suffix}/steered_results_mystery_{mystery_n}.json"
            else:
                file_path = f"results/mystery_steered_300_t_{start_t}_{end_t}_s_{scale}_l_10_{layer}{in_suffix}/steered_results_mystery_{mystery_n}.json"
                file_path = f"results/mystery_steered_300_t_{start_t}_{end_t}_s_{scale}_l_{layer}{in_suffix}/steered_results_mystery_{mystery_n}.json"
        else:
            file_path = f"results/mystery_replaced_st_end_s_{scale}_t_{start_t}_{end_t}_l_0_{layer}_fix_rescale{in_suffix}/steered_results_mystery_{mystery_n}.json"

        print(file_path)
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
            
            raw_llm_answer = raw_llm_answer.strip()

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
        

        # gen_name = "steered_neg_4000" if steering else "replaced-st"
        gen_name = "steered"
        gen_name = gen_name if steered_generation else "original"
        if scale == "0":
            final_dir = Path(f"cot-planning/responses/{formatted_json['domain']}/qwq-32b-{gen_name}-full-300-{start_t}-{end_t}-{scale}{out_suffix}{suffix}/")
        else:
            final_dir = Path(f"cot-planning/responses/{formatted_json['domain']}/qwq-32b-{gen_name}-full-300-{start_t}-{end_t}-{scale}-l-{layer}{out_suffix}{suffix}/")        

        # print(final_dir)

        final_dir.mkdir(parents=True, exist_ok=True)

        with open(final_dir / f"{formatted_json['task']}.json", "w") as f:
            json.dump(formatted_json, f, indent=2)


    main(True)
    main(False)


for i in range(1, 16):
# for i in [14, 15]:
    try:
        ci = None
        for ci in range(3):
            main_gl(i, ci, "sample_avg")
        # main_gl(i, ci, "r")
        # main_gl(i, ci, "rr")
        # main_gl(i, ci, "r")
        # main_gl(i, ci, "mo")
        # main_gl(i, ci, "z")
    except Exception as e:
        print(f"Error processing mystery_{i}: {e}")
        continue