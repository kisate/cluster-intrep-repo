import ray

ray.init(address="auto", namespace="blocksworld")

dataset_actor = ray.get_actor("dataset_actor")

print(ray.get(dataset_actor.get_item_idx.remote(0)))