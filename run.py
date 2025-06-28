import os
import time

from dotenv import load_dotenv
from accelerate import Accelerator, DistributedDataParallelKwargs
from loguru import logger
import wandb

from utils.utils import get_datasets_by_names, get_model_by_names, reformat_args, \
    get_metrics_by_names, load_config_from_yaml_file, get_scenario_by_name, get_loggers_by_names, parse_args, \
    get_text_generation_model_by_name, load_user_simulators, create_user_simulators

from config.config import DatasetConfigForRecommendation
from eval.offline import OfflineEvaluator
from eval.online import OnlineEvaluator
from utils.utils import set_seed
from config.constants import BART_GENERATION, VICUNA, RECOMMENDATION, NEGOTIATION, EMOTIONAL_SUPPORT
#
# from modpl_new.config import ContextualMODPLConfig

from modpl_new_ver2.config import ContextualMODPLConfig
from baselines.PPDPP.config import PPDPPConfig
from baselines.TRIP.config import TRIPConfig
from baselines.DDQL.config import DDQLConfig
from baselines.Envelope.config import EnvelopeConfig
from baselines.DPDP.config import DPDPConfig
from baselines.PADPP_smp.config import SetMaxPADPPConfig
from baselines.PADPP_min_dist.config import MinDistPADPPConfig

# load variables from the .env file
load_dotenv()

if __name__ == '__main__':
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # the current local time
    local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    # parse keywords arguments
    args = parse_args()
    print(vars(args))
    args = reformat_args(vars(args))
    set_seed(args['seed'])

    # initiallize accelerator and device to run the models
    accelerator = Accelerator(device_placement=True, kwargs_handlers=[ddp_kwargs])
    device = accelerator.device

    # construct the scenario
    game_config_file, game_config_class, game_class, game_simulator_class = get_scenario_by_name(args['scenario'])
    game_params = load_config_from_yaml_file(game_config_file)
    game_config = game_config_class(game_params)

    # set the current random seed
    game_config.set_params({
        'seed': args['seed'],
        'model_type': args['model_type'], # type of the llm model,
    })

    # construct a set of datasets.
    dataset_config_classes_and_config_paths = get_datasets_by_names(args['scenario'], args['datasets'])

    # get model config, class and pipeline using model's name
    model_classes_and_pipelines = get_model_by_names(args['scenario'], args['models'])

    # construct metrics
    offline_metrics, online_metrics = get_metrics_by_names(args['scenario'], args['metrics'])

    # construct offline evaluator
    # now we only consider generation-based metrics
    offline_evaluator = OfflineEvaluator(offline_metrics)

    # online evaluator
    online_evaluator = OnlineEvaluator(online_metrics)

    # the package for the text generation model
    generation_packges = get_text_generation_model_by_name(args['scenario'], args['gen_models'])

    # loop overall datasets
    for data_config_path, dataset_class, dataset_scenario_config_class in dataset_config_classes_and_config_paths:

        # create the dataset config
        dataset_params = load_config_from_yaml_file(data_config_path)
        dataset_config = dataset_scenario_config_class(dataset_params)

        # if we are using the recommendation domain
        # we need to the set the domain to the dataset.
        if isinstance(dataset_config, DatasetConfigForRecommendation):
            dataset_config.set_params(
                {"domain": args['domain'],}
            )

        # create the dataset
        dataset = dataset_class(dataset_config)

        # creating the user simulators if it does not exists.
        if not os.path.exists(dataset_config.save_dev_simulator_path) or args['overwrite_sim']:
            # generate user profiles
            train_user_profiles, dev_user_profiles, test_user_profiles = dataset.get_user_profiles()
            logger.info("Creating Dev Set User Simulators ......")
            dev_user_simulators = create_user_simulators(game_simulator_class, dev_user_profiles,
                                                         saved_filed_path=dataset_config.save_dev_simulator_path)
            logger.info("Creating Test User Simulators .....")
            test_user_simulators = create_user_simulators(game_simulator_class, test_user_profiles,
                                                          saved_filed_path=dataset_config.save_test_simulator_path)
        # load the user simulators from file
        else:
            # load the simulator from files
            dev_user_simulators = load_user_simulators(simulator_file_path=dataset_config.save_dev_simulator_path)
            test_user_simulators = load_user_simulators(simulator_file_path=dataset_config.save_test_simulator_path)

        # setting the model type and the flag of using persona
        # according to the model type in the game config class.
        new_dev_user_simulators = []
        for simulator in dev_user_simulators:
            # set the model type
            simulator.set_model_type(game_config.model_type)
            # set the flag of using persona
            simulator.is_using_persona(args['use_persona'])
            new_dev_user_simulators.append(simulator)

        new_test_user_simulators = []
        for simulator in test_user_simulators:
            # set the model type
            simulator.set_model_type(game_config.model_type)
            # set the flag of using persona
            simulator.is_using_persona(args['use_persona'])
            new_test_user_simulators.append(simulator)

        # construct the pipeline for each model and run the pipeline on a specific scenario
        for (
                config_file, config_class, model_class, pipeline_class, trainer_class) in model_classes_and_pipelines:
            # create the game
            game = game_class(game_config=game_config, dataset_config=dataset_config)

            # load and creata the model config
            model_params = load_config_from_yaml_file(config_file)
            model_config = config_class(model_params)

            # set the llm model type
            # only affects prompting method
            model_config.set_params(
                {'model_type': game_config.model_type,
                 }
            )

            # assign the number of goals, topics to the model config
            if args['scenario'] == RECOMMENDATION:
                model_config.set_params(
                    {'n_goals': dataset.n_goals,
                     'n_topics': dataset.n_topics,
                     'n_objectives': game_config.n_objectives,
                     'device': device,
                     'scenario_name': RECOMMENDATION,
                     'domain': args['domain']
                     }
                )
            # for negotiation, we only have the goals
            elif args['scenario'] == NEGOTIATION:
                model_config.set_params(
                    {'n_goals': dataset.n_goals,
                     'n_objectives': game_config.n_objectives,
                     'device': device,
                     'scenario_name': NEGOTIATION,
                     }
                )
            # similarly for emotional support conversation
            # we only have the goals
            elif args['scenario'] == EMOTIONAL_SUPPORT:
                model_config.set_params(
                    {
                        'n_goals': dataset.n_goals,
                        'n_objectives': game_config.n_objectives,
                        'device': device,
                        'scenario_name': EMOTIONAL_SUPPORT
                    }
                )
            # For ablation study only
            # we only consider the MODPL model
            if args['ablation'] is not None:
                if isinstance(model_config, ContextualMODPLConfig) or isinstance(model_config, EnvelopeConfig) or isinstance(model_config, MinDistPADPPConfig):
                    model_config.set_params(
                        {
                            'ablation': args['ablation'], # no rl tuning
                            'use_gpi': args['use_gpi'], # no gpi
                            'num_test_cases': args['num_test_cases'], # number of test cases
                            "n_preferences": args["n_preferences"] # number of preferences
                        }
                    )
                    
            # we only consider the MODPL model
            if isinstance(model_config, ContextualMODPLConfig) or isinstance(model_config, PPDPPConfig) or isinstance(model_config, TRIPConfig) \
                or isinstance(model_config, DDQLConfig) or isinstance(model_config, EnvelopeConfig) or isinstance(model_config, DPDPConfig) or \
                    isinstance(model_config, SetMaxPADPPConfig) or isinstance(model_config, MinDistPADPPConfig):
                
                # For weight-specific inference
                # Adaptability experiments
                if args['objective_weight'] is not None and (isinstance(model_config, ContextualMODPLConfig) or isinstance(model_config, EnvelopeConfig)) or isinstance(model_config, MinDistPADPPConfig):
                    
                    print(args['objective_weight'])
                    # convert objective weights to float format
                    objective_weight = [float(x) for x in args['objective_weight']]
                    
                    # set the new objective weight
                    model_config.set_params(
                        {
                            'objective_weight': objective_weight
                        }
                    )
                    
                # For objective-specific inference, e.g sl_ratio, fairness, sr, etc.
                # default is uniform
                # set the numbef of training RL epochs to control the number of training episodes.
                # default 50.
                model_config.set_params(
                    {
                        'prioritized_objective': args['prioritized_objective'],
                        'test_phase': args['test_phase'],
                        'num_train_rl_epochs': args['num_train_rl_epochs'],
                    }
                )

            # construct model
            model = model_class(model_config)
            model_name = str(model.__class__.__name__)

            # loop over different text generation model
            # for each text generation model construct a specific policy pipeline
            for gen_name, generation_package in list(zip(args['gen_models'].split(','), generation_packges)):
                # if the generation method is bart
                if gen_name.strip() == BART_GENERATION:
                    generation_config_path, generation_config_class, generation_model_class, generation_trainer_class, generation_pipeline_class, generation_class = generation_package
                    # load generation parameters from file
                    # including model parameters and parameters for inference
                    generation_params = load_config_from_yaml_file(generation_config_path)
                    generation_config = generation_config_class(generation_params)

                    # assign the generation model to the current devices
                    # make sure the generation model use appropriate special token dictionary
                    generation_config.set_params(
                        {
                            'device': device,
                            'special_tokens_dict': model_config.special_tokens_dict
                        }
                    )

                    # construct the generation model
                    generation_model = generation_model_class(generation_config)
                    generation_model_name = str(generation_model.__class__.__name__)

                    # construct the generation trainer
                    generation_trainer = generation_trainer_class(dataset_config=dataset_config,
                                                                  accelerator=accelerator,
                                                                  game_config=game_config,
                                                                  model_config=generation_config,
                                                                  game=game,
                                                                  model=generation_model,
                                                                  offline_evaluator=offline_evaluator,
                                                                  online_evaluator=None,
                                                                  loggers=None)

                    # create the generation pipeline
                    generation_pipeline = generation_pipeline_class(dataset_config=dataset_config,
                                                                    dataset=dataset,
                                                                    trainer=generation_trainer,
                                                                    )

                    # create generation method
                    # since we use the generation for policy training, then we assign the flag is_test = True
                    generation_method = generation_class(generation_config, generation_pipeline, is_test=True)
                # chatgpt, vicuna for response generation
                else:
                    generation_config_path, generation_prompt, generation_config_class, generation_class = generation_package

                    # load the configurations from file
                    # i.e max gen length or temperature
                    generation_params = load_config_from_yaml_file(generation_config_path)

                    # construct the generation config
                    generation_config = generation_config_class(generation_params)

                    # set the prompt for the generation config
                    generation_config.set_params(
                        {
                            'prompt': generation_prompt,
                            "scenario_name": game_config.name,
                            "dataset": dataset_config.dataset_name
                        }
                    )
                    # the other parameters to the vicuna model
                    if gen_name == VICUNA:
                        generation_config.set_params(
                            {
                                "device": args["device"],
                                "num_gpus": args['num_gpus'],
                                "load_8bit": args['load_8bit'],
                                "cpu_offloading": args['cpu_offloading'],
                                "max_gpu_memory": args["max_gpu_memory"]
                            }
                        )

                    # construct the generation method
                    generation_method = generation_class(generation_config, None, None)

                # construct loggers
                loggers = get_loggers_by_names(args['loggers'],
                                               game_config=game_config,
                                               dataset_config=dataset_config,
                                               model_config=model_config,
                                               local_time=local_time,
                                               random_seed=args['seed'],
                                               model_name=model_name,
                                               exp_name=args['exp_name'],
                                               log_dir=args['log_dir'],
                                               wandb_key=os.getenv("WANDB_KEY"),
                                               project_name=args['project_name']
                                               )

                # construct the trainer
                trainer = trainer_class(accelerator=accelerator,
                                        game_config=game_config,
                                        model_config=model_config,
                                        game=game,
                                        model=model,
                                        generation_method=generation_method,
                                        offline_evaluator=offline_evaluator,
                                        online_evaluator=online_evaluator,
                                        loggers=loggers
                                        )

                # create the pipeline
                pipeline = pipeline_class(
                    dataset_config=dataset_config,
                    dataset=dataset,
                    trainer=trainer,
                )

                # load user simulators for rl training and online evaluation
                pipeline.set_user_simulators(
                    # run the pipeline with 1 simulators
                    # dev_simulators=[dev_user_simulators[0]],
                    # test_simulators=[test_user_simulators[0]],
                    # run the pipeline with multiple simulators
                    dev_simulators=new_dev_user_simulators,
                    test_simulators=new_test_user_simulators
                )

                # computing the execution time
                start_time = time.time()

                # prepare the pipeline
                pipeline.execute()

                end_time = time.time() - start_time

                # print the execution time
                logger.info(f"Execution time: {end_time} seconds")

                # stop wandb to create a new run
                wandb.finish()
