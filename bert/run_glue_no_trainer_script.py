import os
import argparse
import json


def glue_main(config):
    pre_trained_models_dir_list = config['pre_trained_models_dir']
    if config["pre_trained_models_root"] is not None:
        for root, dirs, files in os.walk(config["pre_trained_models_root"]):
            for dir in dirs:
                if "run_" in dir:
                    pre_trained_models_dir_list.append(os.path.join(root, dir))

    task_name = config['task_name']
    save_dir = config['save_dir']
    run_name = config['run_name']
    run_type = config['run_type']
    save_model = config['save_model']
    train_frac = config['train_frac'] if 'train_frac' in config else 1.0
    seed = config['seed']

    if run_name is not None:
        output_dir = os.path.join(save_dir, task_name, f'{run_name}_trainfrac_{train_frac}_seed_{seed}')
    else:
        output_dir = os.path.join(save_dir, f'{task_name}', f'trainfrac_{train_frac}_seed_{seed}')
    print(f'Processing {output_dir}...')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.system(f"python -m run_glue \
                           --output_dir {output_dir} \
                           --task_name {task_name} \
                           --model_name_or_path=bert-base-uncased \
                           --max_length 128 \
                           --per_device_train_batch_size 32 \
                           --per_device_eval_batch_size 32 \
                           --learning_rate 2e-5 \
                           --gradient_accumulation_steps 2 \
                           --num_train_epochs {config['epochs']} \
                           --seed={seed}")

        with open(os.path.join(output_dir, 'exp_cfg.json'), 'w') as json_file:
            json.dump(config, json_file)
    else:
        print(f'Skipping {output_dir}... already exists')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    glue_main(config)