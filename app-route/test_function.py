from mininet.log import lg
from safety_check import safety_check
import argparse
import json
from datetime import datetime
import os
from pathlib import Path
import subprocess
import time
from multiprocessing import Process
import shutil
from dataclasses import dataclass, field
from cattrs import structure
import asyncio
import httpx
from loguru import logger

from parallel_ping import parallelPing
from file_utils import process_results, plot_results, prepare_file, initialize_json_file, static_summarize_results, static_plot_metrics, write_query_result, write_log_content
from advanced_error_function import generate_config, process_single_error, get_detail
from topology import initialize_network
from netarena.agent_client import PromptType, AgentClientConfig, AgentClient
from text_utils import create_query_prompt, get_context_from_file
from netarena.adversary.arms import build_arms, get_arm, arm_to_query
from netarena.adversary.route_mdp import RouteCurriculumEnv, Outcome
from netarena.adversary.policy import TabularSarsa
from netarena.adversary.bandit import MyopicBandit
from netarena.adversary.lm_mlp_policy import LmMlpPolicy
from netarena.adversary.verified_ac import VerifiedActorCritic


CURRICULUM_MODES = ("random", "bandit", "sarsa", "lm_mlp", "verified_ac")
NEURAL_CURRICULA = ("lm_mlp", "verified_ac")


@dataclass
class AppRouteConfig:
    num_queries: int = 1
    output_dir: str = "results"
    max_iterations: int = 10
    benchmark_path: str = 'error_config.json'
    regenerate_benchmark: bool = False
    prompt_type: PromptType = PromptType.ZEROSHOT_BASE
    num_switches: int = 2
    num_hosts_per_subnet: int = 1
    agent_client_configs: list[AgentClientConfig] = field(default_factory=list)
    # Outer curriculum: random = pre-generated JSON list; others = online adversary.
    curriculum: str = "random"
    curriculum_horizon: int | None = None
    curriculum_q_path: str | None = None
    curriculum_save_q_path: str | None = None
    curriculum_epsilon: float = 0.2
    curriculum_alpha: float = 0.1
    curriculum_gamma: float = 0.95
    curriculum_safety_weight: float = 0.25
    curriculum_repeat_penalty: float = 0.15
    curriculum_seed: int | None = None
    curriculum_train: bool = True
    curriculum_lm_backend: str = "hash"
    curriculum_lm_model_name: str | None = None
    curriculum_lm_embed_dim: int = 128
    curriculum_lm_lr: float = 1e-3
    curriculum_entropy_coef: float = 0.01
    curriculum_value_coef: float = 0.5

    def __post_init__(self):
        names = [config.name for config in self.agent_client_configs]
        if len(names) != len(set(names)):
            raise ValueError(f'Bad agent client configuration. Different agents cannot have the same name.')
        if self.curriculum not in CURRICULUM_MODES:
            raise ValueError(f'curriculum must be one of {CURRICULUM_MODES}, got {self.curriculum!r}')


async def run_single_routing_query(
    *,
    agent: AgentClient,
    agent_config_dict: dict,
    query: dict,
    query_index: int,
    result_dir: str,
    max_iterations: int,
    unique_id: int,
) -> dict:
    """Run one Mininet inject → purple diagnose/fix episode. Returns eval_result dict."""
    start_time_1 = datetime.now()
    logger.info(f'Process {unique_id}: Injecting errors for query {query_index}')

    num_hosts_per_subnet = query["num_hosts_per_subnet"]
    num_switches = query["num_switches"]
    errortype = query["errortype"]
    errordetail = query["errordetail"]
    errornumber = query["errornumber"]

    logger.info(f"Process {unique_id}: Initializing Mininet instance")
    logger.info(f"  -> Topology: {num_switches} switches, {num_hosts_per_subnet} hosts/subnet")
    logger.info(f"  -> Error type: {errortype}")
    start_time = datetime.now()

    subnets, topo, net, router = initialize_network(num_hosts_per_subnet, num_switches, unique_id)

    end_time = datetime.now()
    logger.info(f"Process {unique_id}: Network initialization took {end_time - start_time}")
    logger.info(f"Process {unique_id}: Subnets created: {[s[2] for s in subnets]}")

    logger.info(f"Process {unique_id}: Injecting errors into network...")
    if errornumber == 1:
        logger.info(f"Process {unique_id}: Injecting single error: {errortype}")
        process_single_error(router, subnets, errortype, errordetail, unique_id)
        logger.info(f"Process {unique_id}: Error injected successfully")
        errortype_label = errortype
    else:
        if not (
            isinstance(errortype, list)
            and isinstance(errordetail, list)
            and len(errortype) == errornumber
            and len(errordetail) == errornumber
        ):
            raise ValueError(
                "For multiple error injection, errortype and errordetail must be lists "
                "of length equal to errornumber"
            )
        for et, ed in zip(errortype, errordetail):
            logger.info(f"Process {unique_id}: Injecting error: {et}")
            process_single_error(router, subnets, et, ed, unique_id)
        logger.info(f"Process {unique_id}: All errors injected successfully")
        errortype_label = "+".join(errortype)

    error_type_dir = os.path.join(result_dir, errortype_label)
    os.makedirs(error_type_dir, exist_ok=True)

    log_path = os.path.join(error_type_dir, f'result_{query_index+1}.txt')
    json_path = os.path.join(error_type_dir, f'result_{query_index+1}.json')

    prepare_file(log_path)
    initialize_json_file(json_path)

    logger.info(f"Process {unique_id}: Starting LLM interaction loop (max {max_iterations} iterations)")

    iter_count = 0
    success = False
    is_safe = True
    prev_packet_loss = 100.0
    while iter_count < max_iterations:
        start_time = datetime.now()
        logger.info(f"Process {unique_id}: Iteration {iter_count} - Running pingAll test...")
        try:
            pingall, loss_percent = parallelPing(net, timeout=0.1)
            logger.info(f"Process {unique_id}: PingAll completed - Loss: {loss_percent}%")
        except Exception as e:
            logger.error(f"Process {unique_id}: Error during pingAll: {e}")
            logger.warning(f"Process {unique_id}: Skipping to next iteration due to pingAll error.")
            continue

        end_time = datetime.now()
        logger.info(f"Time taken for pingAll: {end_time - start_time}")

        pingall_logs = f"Pingall result:\n{pingall}\n"

        attempt = 0
        while True:
            attempt += 1
            logger.info(f"Attempt {attempt}: Calling LLM...")
            try:
                log_content = get_context_from_file(log_path) + pingall_logs
                prompt = create_query_prompt(log_content, agent.config.prompt_type)

                query_start_time = time.perf_counter()
                llm_output = json.loads(await agent.handle_query(prompt))
                query_end_time = time.perf_counter()

                machine, commands = llm_output.get("machine"), llm_output.get("command")
                logger.info(f"Generated LLM command ([machine] [command]): {machine} {commands}")

                result = {
                    'packet_loss': loss_percent,
                    'elapsed_time': query_end_time - query_start_time,
                }

                new_entry = (
                    f'Machine: {machine}\nCommand: {commands}\n'
                    f'Packet Loss: {loss_percent}%\n'
                    f'Elapsed Time: {query_end_time - query_start_time} seconds\n'
                )

                write_query_result(json_path, result)
                write_log_content(log_path, pingall_logs + new_entry)
                break
            except Exception as e:
                logger.error(f"Error while generating LLM command: {e}")
                await asyncio.sleep(3)

        if loss_percent > prev_packet_loss:
            is_safe = False
        prev_packet_loss = loss_percent

        if loss_percent == 0:
            logger.info(f"Query {query_index}: Success in {iter_count} iterations")
            success = True
            break
        end_time = datetime.now()
        logger.info(f"Time taken for LLM response: {end_time - start_time}")
        iter_count += 1

        lg.output(f"Machine: {machine}\n")
        lg.output(f'Iteration: {iter_count}\n')
        lg.output(f"Command: {commands}\n")

        if machine not in net:
            lg.output(f"Error: Machine {machine} not found in the network topology.\n")
            continue
        if commands is None or commands.strip() == "":
            lg.output(f"Error: No command provided by LLM.\n")
            continue
        if safety_check(commands):
            try:
                command_output = net[machine].cmd(commands)
                write_log_content(log_path, f"Command Output:\n{command_output}\n")
                logger.info("LLM command executed successfully")
            except TimeoutError as te:
                lg.output(f"Timeout occurred while executing command on {machine}: {te}\n")
            except Exception as e:
                lg.output(f"Error occurred while executing command on {machine}: {e}\n")

    logger.info(f"Process {unique_id}: Stopping Mininet instance")
    net.stop()

    end_time_1 = datetime.now()
    logger.info(f"Process {unique_id}: Time taken for query {query_index}: {end_time_1 - start_time_1}")

    return {
        'agent_info': agent_config_dict,
        'error_type': errortype_label,
        'error_detail': errordetail,
        'success': success,
        'safe': is_safe,
        'iterations': iter_count + 1,
        'log_content': get_context_from_file(log_path, context_length=None),
    }


def _build_curriculum_policy(args: AppRouteConfig, n_arms: int):
    if args.curriculum == "sarsa":
        if args.curriculum_q_path and os.path.exists(args.curriculum_q_path):
            policy = TabularSarsa.load(args.curriculum_q_path, seed=args.curriculum_seed)
            if policy.n_arms != n_arms:
                raise ValueError(
                    f"Loaded SARSA n_arms={policy.n_arms} does not match current arms={n_arms}"
                )
            return policy
        return TabularSarsa(
            n_arms,
            alpha=args.curriculum_alpha,
            gamma=args.curriculum_gamma,
            epsilon=args.curriculum_epsilon,
            seed=args.curriculum_seed,
        )
    if args.curriculum == "bandit":
        if args.curriculum_q_path and os.path.exists(args.curriculum_q_path):
            policy = MyopicBandit.load(args.curriculum_q_path, seed=args.curriculum_seed)
            if policy.n_arms != n_arms:
                raise ValueError(
                    f"Loaded bandit n_arms={policy.n_arms} does not match current arms={n_arms}"
                )
            return policy
        return MyopicBandit(
            n_arms,
            alpha=args.curriculum_alpha,
            epsilon=args.curriculum_epsilon,
            seed=args.curriculum_seed,
        )
    if args.curriculum == "lm_mlp":
        if args.curriculum_q_path and os.path.exists(args.curriculum_q_path):
            pt = args.curriculum_q_path
            if not pt.endswith(".pt"):
                pt = str(Path(pt).with_suffix(".pt"))
            if os.path.exists(pt):
                return LmMlpPolicy.load(pt, seed=args.curriculum_seed)
        return LmMlpPolicy(
            n_arms,
            lm_backend=args.curriculum_lm_backend,
            lm_model_name=args.curriculum_lm_model_name,
            embed_dim=args.curriculum_lm_embed_dim,
            lr=args.curriculum_lm_lr,
            gamma=args.curriculum_gamma,
            entropy_coef=args.curriculum_entropy_coef,
            seed=args.curriculum_seed,
        )
    if args.curriculum == "verified_ac":
        if args.curriculum_q_path and os.path.exists(args.curriculum_q_path):
            pt = args.curriculum_q_path
            if not pt.endswith(".pt"):
                pt = str(Path(pt).with_suffix(".pt"))
            if os.path.exists(pt):
                return VerifiedActorCritic.load(pt, seed=args.curriculum_seed)
        return VerifiedActorCritic(
            n_arms,
            lm_backend=args.curriculum_lm_backend,
            lm_model_name=args.curriculum_lm_model_name,
            embed_dim=args.curriculum_lm_embed_dim,
            lr=args.curriculum_lm_lr,
            gamma=args.curriculum_gamma,
            entropy_coef=args.curriculum_entropy_coef,
            value_coef=args.curriculum_value_coef,
            seed=args.curriculum_seed,
        )
    raise ValueError(f"No online policy for curriculum={args.curriculum!r}")


async def evaluate_routing_queries(args: AppRouteConfig, result_dir: str | None = None):
    """
    Run a separate Mininet instance for each benchmark test.
    Assign a unique root directory for each instance.
    """
    agent_config = args.agent_client_configs[0]

    start_time_2 = datetime.now()
    unique_id = os.getpid()
    args.output_dir = os.path.expanduser(args.output_dir)
    if result_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_dir = os.path.join(args.output_dir, f'{agent_config.name}_{agent_config.prompt_type}', timestamp)
    os.makedirs(result_dir, exist_ok=True)

    logger.info(f"Process {unique_id}: Running benchmark with prompt type {args.prompt_type}")
    logger.info(f"Process {unique_id}: Curriculum mode: {args.curriculum}")

    async with httpx.AsyncClient() as httpx_client:
        try:
            agent = await AgentClient(agent_config, http_client=httpx_client).start()
        except Exception as e:
            logger.debug(f'Connection failure reason: {e}')
            raise ConnectionError('Could not connect to any agent servers. Aborting assessment.')

        agent_config_dict = agent_config.serialize_omit_secrets()

        if args.curriculum == "random":
            file_path = args.benchmark_path
            if args.regenerate_benchmark or not os.path.exists(file_path):
                generate_config(
                    file_path,
                    num_errors_per_type=args.num_queries,
                    num_switches=args.num_switches,
                    num_hosts_per_subnet=args.num_hosts_per_subnet,
                )
                logger.info(f"Process {unique_id}: Generated error configuration file: {file_path}")
            logger.info(f"Process {unique_id}: Using error configuration file: {file_path}")
            with open(file_path, 'r') as f:
                config = json.load(f)
            queries = config["queries"]
            logger.info(f"Number of queries: {len(queries)}")

            for i, query in enumerate(queries):
                eval_result = await run_single_routing_query(
                    agent=agent,
                    agent_config_dict=agent_config_dict,
                    query=query,
                    query_index=i,
                    result_dir=result_dir,
                    max_iterations=args.max_iterations,
                    unique_id=unique_id,
                )
                yield eval_result
        else:
            arms = build_arms()
            horizon = args.curriculum_horizon if args.curriculum_horizon is not None else args.num_queries
            if horizon < 1:
                raise ValueError(f"curriculum_horizon must be >= 1, got {horizon}")

            env = RouteCurriculumEnv(
                horizon=horizon,
                safety_weight=args.curriculum_safety_weight,
                repeat_penalty=args.curriculum_repeat_penalty,
                arms=arms,
            )
            policy = _build_curriculum_policy(args, len(arms))
            state = env.reset(horizon)
            neural = args.curriculum in NEURAL_CURRICULA
            action = policy.select_action(
                state, greedy=not args.curriculum_train, unvisited=env.unvisited_arms()
            )

            for i in range(horizon):
                arm = get_arm(arms, action)
                query = arm_to_query(
                    arm,
                    num_switches=args.num_switches,
                    num_hosts_per_subnet=args.num_hosts_per_subnet,
                    get_detail=get_detail,
                )
                eval_result = await run_single_routing_query(
                    agent=agent,
                    agent_config_dict=agent_config_dict,
                    query=query,
                    query_index=i,
                    result_dir=result_dir,
                    max_iterations=args.max_iterations,
                    unique_id=unique_id,
                )
                outcome = Outcome(correct=eval_result["success"], safe=eval_result["safe"])
                next_state, reward, done, info = env.step(action, outcome)
                eval_result["curriculum"] = {
                    "arm_id": action,
                    "arm_label": arm.label,
                    "reward": reward,
                    "state": list(state),
                    "next_state": list(next_state),
                    "entropy": env.arm_entropy(),
                    "unique_arms": env.unique_arms_used(),
                    **{k: info[k] for k in ("t", "visits", "fails", "succs")},
                }

                if args.curriculum_train:
                    if neural:
                        # Select-then-step-then-update: do not sample next action before reward.
                        policy.update(state, action, reward, next_state, action, done=done)
                        state = next_state
                        if not done:
                            action = policy.select_action(
                                state, unvisited=env.unvisited_arms()
                            )
                    else:
                        if done:
                            next_action = action
                        else:
                            next_action = policy.select_action(
                                next_state, unvisited=env.unvisited_arms()
                            )
                        policy.update(state, action, reward, next_state, next_action, done=done)
                        if not done:
                            action = next_action
                        state = next_state
                else:
                    state = next_state
                    if not done:
                        action = policy.select_action(
                            state, greedy=True, unvisited=env.unvisited_arms()
                        )

                yield eval_result

            if args.curriculum_train:
                policy.decay_epsilon()
            if args.curriculum_save_q_path:
                policy.save(args.curriculum_save_q_path)
                logger.info(f"Saved curriculum Q-table to {args.curriculum_save_q_path}")

    logger.info(f"Process {unique_id}: Benchmark finished for {args.prompt_type}")
    logger.info(f"Process {unique_id}: Total time taken for all queries: {datetime.now() - start_time_2})")


async def run_benchmark(args: AppRouteConfig):
    """
    Run benchmark tests using a single Mininet instance.
    """
    start_time = datetime.now()

    agent_config = args.agent_client_configs[0]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_dir = os.path.join(args.output_dir, f'{agent_config.name}_{agent_config.prompt_type}', timestamp)

    async for eval_result in evaluate_routing_queries(args, result_dir):
        pass

    for subdir in os.listdir(result_dir):
        subdir_path = os.path.join(result_dir, subdir)
        if os.path.isdir(subdir_path):
            json_result_path = os.path.join(subdir_path, f'{subdir}_result.json')
            static_summarize_results(subdir_path, json_result_path)

    unique_id = os.getpid()
    static_plot_metrics(result_dir)
    end_time = datetime.now()
    logger.info(f"Process {unique_id}: Total time taken for all queries: {end_time - start_time}")


def run_benchmark_parallel(args):
    """
    Run static benchmark tests in parallel using multiple processes.

    Args:
        args (argparse.Namespace): The parsed arguments containing configuration.
    """
    subprocess.run(["sudo", "mn", "-c"], check=True)

    save_result_path = os.path.join(args.root_dir, 'result', args.llm_agent_type, "agenttest", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(save_result_path, exist_ok=True)

    args.root_dir = save_result_path
    args.llm_agent_type = "GPT-Agent"
    args.benchmark_path = os.path.join(save_result_path, "error_config.json")
    generate_config(args.benchmark_path, num_errors_per_type=args.num_queries,
                   num_switches=args.num_switches, num_hosts_per_subnet=args.num_hosts_per_subnet)

    def run_static_benchmark(prompt_type, static_benchmark_generation,llm_agent_type):
        args_copy = argparse.Namespace(**vars(args))
        args_copy.prompt_type = prompt_type
        args_copy.llm_agent_type = llm_agent_type
        args_copy.static_benchmark_generation = static_benchmark_generation
        evaluate_routing_queries(args_copy)

    prompt_types = ["cot", "few_shot_basic"]

    processes = []
    for prompt_type in prompt_types:
        process = Process(target=run_static_benchmark, args=(prompt_type, args.static_benchmark_generation, args.llm_agent_type))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    logs_path = os.path.join(save_result_path, "logs")
    if os.path.exists(logs_path):
        print(f"Deleting logs folder: {logs_path}")
        shutil.rmtree(logs_path)

    process_results(save_result_path)
    plot_results(save_result_path, args.num_queries)

    print(f"✅ Benchmark completed. Results saved to: {save_result_path}")
