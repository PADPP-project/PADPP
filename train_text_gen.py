import os
import time

from dotenv import load_dotenv
from accelerate import Accelerator

from utils.utils import get_datasets_by_names, reformat_args, \
    get_metrics_by_names

from utils.utils import parse_args, load_config_from_yaml_file, get_scenario_by_name, get_text_generation_model_by_name
from eval.offline import OfflineEvaluator

from base.text_gen import PLMGeneration
from utils.utils import get_loggers_by_names

from config.constants import RECOMMENDATION, NEGOTIATION, rec_special_tokens_dict, neg_special_tokens_dict, \
    EMOTIONAL_SUPPORT, es_special_tokens_dict

# load variables from the .env file
load_dotenv()

if __name__ == '__main__':
    # the current local time
    local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    # parse keywords arguments
    args = parse_args()
    print(vars(args))
    args = reformat_args(vars(args))

    # initiallize accelerator and device to run the models
    accelerator = Accelerator(device_placement=False)
    device = accelerator.device

    # construct the scenario
    game_config_file, game_config_class, game_class, _ = get_scenario_by_name(args['scenario'])
    game_params = load_config_from_yaml_file(game_config_file)
    game_config = game_config_class(game_params)

    # set the current random seed
    game_config.set_params({
        'seed': args['seed']
    })
    # construct a set of datasets.
    dataset_config_classes_and_config_paths = get_datasets_by_names(args['scenario'], args['datasets'])

    # get model config, class and pipeline using model's name
    generation_packges = get_text_generation_model_by_name(args['scenario'], args['models'])

    # construct metrics
    offline_metrics, online_metrics = get_metrics_by_names(args['scenario'], args['metrics'])

    # construct offline evaluator
    offline_evaluator = OfflineEvaluator(offline_metrics, policy_eval=False)

    # loop overall datasets
    for data_config_path, dataset_class, dataset_scenario_config_class in dataset_config_classes_and_config_paths:

        # create the dataset config
        dataset_params = load_config_from_yaml_file(data_config_path)
        dataset_config = dataset_scenario_config_class(dataset_params)

        # create the dataset
        dataset = dataset_class(dataset_config)

        for generation_config_path, generation_config_class, generation_model_class, generation_trainer_class, generation_pipeline_class, generation_class in generation_packges:

            # do not consider LLM generation
            if not issubclass(generation_class, PLMGeneration):
                continue

            game = game_class(game_config=game_config, dataset_config=dataset_config)

            # load generation parameters from file
            # including model parameters and parameters for inference
            generation_params = load_config_from_yaml_file(generation_config_path)
            generation_config = generation_config_class(generation_params)

            # seting the special tokens dict
            # the recommendation scenario
            if args['scenario'] == RECOMMENDATION:
                special_tokens_dict = rec_special_tokens_dict
            # the negotiation scenario
            elif args['scenario'] == NEGOTIATION:
                special_tokens_dict = neg_special_tokens_dict
            elif args['scenario'] == EMOTIONAL_SUPPORT:
                special_tokens_dict = es_special_tokens_dict
            else:
                raise Exception("Invalid Scenario .....")

            generation_config.set_params(
                {
                    'special_tokens_dict': special_tokens_dict
                }
            )
            # construct the generation model
            generation_model = generation_model_class(generation_config)
            generation_model_name = str(generation_model.__class__.__name__)

            # construct loggers
            loggers = get_loggers_by_names(args['loggers'],
                                           game_config=game_config,
                                           dataset_config=dataset_config,
                                           model_config=generation_config,
                                           local_time=local_time,
                                           random_seed=args['seed'],
                                           model_name=generation_model_name,
                                           log_dir=args['log_dir'],
                                           wandb_key=os.getenv("WANDB_KEY"),
                                           project_name=args['project_name'])

            # construct the generation trainer
            generation_trainer = generation_trainer_class(dataset_config=dataset_config,
                                                          accelerator=accelerator,
                                                          game_config=game_config,
                                                          model_config=generation_config,
                                                          game=game,
                                                          model=generation_model,
                                                          offline_evaluator=offline_evaluator,
                                                          online_evaluator=None,
                                                          loggers=loggers)

            # create the generation pipeline
            generation_pipeline = generation_pipeline_class(dataset_config=dataset_config,
                                                            dataset=dataset,
                                                            trainer=generation_trainer)

            # generation method
            generation_method = generation_class(generation_config, generation_pipeline)

            # prepare the generation method on the current dataset
            generation_method.prepare()
