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
    seed = config['seed']
    train_strategy = config['train_strategy']
    specialize_after = config['specialize_after']

    if run_name is not None:
        output_dir = os.path.join(save_dir, task_name, f'{run_name}_{train_strategy}_{specialize_after}_seed_{seed}')
        run_name = f'{task_name}_{run_name}_{train_strategy}_{specialize_after}'
    else:
        output_dir = os.path.join(save_dir, f'{task_name}', f'_{train_strategy}_{specialize_after}_seed_{seed}')
        run_name = f'{task_name}_{train_strategy}_{specialize_after}'
    print(f'Processing {output_dir}...')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    os.system(f"python -m run_glue \
                           --output_dir {output_dir} \
                           --task_name {task_name} \
                           --model_name_or_path=bert-base-uncased \
                           --run_name {run_name} \
                           --evaluation_strategy=steps \
                           --logging_dir {output_dir} \
                           --max_seq_length 128 \
                           --do_train \
                           --save_strategy no \
                           --fp16 \
                           --do_eval \
                           --eval_steps {config['eval_steps']} \
                           --logging_steps 10 \
                           --per_device_train_batch_size 32 \
                           --per_device_eval_batch_size 32 \
                           --learning_rate {config['learning_rate']} \
                           --lr_scheduler_type {config['lr_scheduler_type']} \
                           --gradient_accumulation_steps 2 \
                           --num_train_epochs {config['epochs']} \
                           --seed={seed} \
                           --train_strategy={train_strategy} \
                           --specialize_after={specialize_after} \
                           --overwrite_output_dir")

    with open(os.path.join(output_dir, 'exp_cfg.json'), 'w') as json_file:
        json.dump(config, json_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    for epochs in [2]:
        for learning_rate in [5e-5]:
            for seed in [3]:
                for strategy in ["specialize_gradually_loss_fn", "specialize_gradually_consume_and_pass"]:
                    config['epochs'] = epochs
                    config['learning_rate'] = learning_rate
                    config['seed'] = seed
                    config['train_strategy'] = strategy
                    config['run_name'] = f'{epochs}_{learning_rate}'
                    glue_main(config)

    # glue_main(config)

    #"baseline", "specialize", "specialize_gradually_loss_fn", "specialize_gradually_consume_and_pass"
