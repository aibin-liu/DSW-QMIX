from config.arguments import get_arg
from envs import REGISTRY as env_REGISTRY
from modules.agents import REGISTRY as agent_REGISTRY
from learners import REGISTRY as learner_REGISTRY
from databuffers.replaybuffer import ReplayBuffer, EpisodeReplayBuffer
from components.epsilon_schedules import DecayThenFlatSchedule
from utils.utils import cleanup_dir

import torch as th
import numpy as np
import os
import pickle

if __name__ == '__main__':
    args = get_arg()

    # init logging directory and logging file
    log_dir = os.path.expanduser("./log/" + args.log_dir)
    cleanup_dir(log_dir)
    eval_log_dir = log_dir + "/eval"
    cleanup_dir(eval_log_dir)
    model_log_dir = log_dir + "/model"
    cleanup_dir(model_log_dir)
    
    global_reward_file = open("%s/global_reward.log" % (log_dir), "w", 1)
    loss_file = open("%s/loss.log" % (log_dir), "w", 1)
    if args.rl_model == "rnn":
        tderror_loss_file = open("%s/tderror_loss.log" % (log_dir), "w", 1)
        mono_loss_file = open("%s/mono_loss.log" % (log_dir), "w", 1)
    env_file = open("%s/env.pickle" % (log_dir), "wb", 1)

    # setup/load Environment
    if args.env_path == None:
        env = env_REGISTRY[args.application](args)
        n_agent, state_space, observation_spaces, action_spaces, n_opponent_actions = env.setup() # obses: observation states for the agents; state: the global state for the mixer
    else:
        load_env_file = open(args.env_path, "rb")
        env = pickle.load(load_env_file)
        load_env_file.close()
        n_agent, state_space, observation_spaces, action_spaces, n_opponent_actions = env.get_rlinfo()
    # save the image of the environment
    pickle.dump(env, env_file)
    env_file.close()
    args.state_dim = state_space
    args.n_opponent_actions = n_opponent_actions 
    
    # init env parameters
    env.set_logger(log_dir)
    env.init_training()
    env.set_scheme(args.rl_model)

    # init replay buffer
    if args.rl_model == "rnn":
        buffer = EpisodeReplayBuffer(args.buffer_size)
    else:
        buffer = ReplayBuffer(args.buffer_size)

    
    # init epsilon decay schedule
    schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, int(args.training_epochs * 0.9), decay=args.epsilon_scheduler)

    # init agents
    agents = []
    for i in range(n_agent):
        if args.rl_model == 'simple':
            agent = agent_REGISTRY['simple'](observation_spaces[i], action_spaces[i])
        else:
            agent = agent_REGISTRY['rnn'](observation_spaces[i], action_spaces[i])
        
        agents.append(agent)

    # init learner
    learner = learner_REGISTRY[args.rl_model](agents, args)
    if args.model_load_path != None:
        learner.load_models(args.model_load_path)
    
    for episode in range(args.training_episodes):
        buffer.clear()
        for epoch in range(args.training_epochs):
            state, obses = env.reset()
            epoch_episode = []
            rnn_hiddens = [None] * n_agent
            for env_t in range(args.max_env_t):
                print("episode:{} epoch:{} step: {}".format(episode, epoch, env_t))
                transition_data = {'states': obses, 'global_state': state}
                # make action (rnn/DSW: joint argmax of Q_r + lambda_p Q_p - (rho/2) relu(-Q_p)^2)
                with th.no_grad():
                    dev = next(agents[0].parameters()).device
                    qs_list = []
                    for i in range(n_agent):
                        if args.rl_model == "rnn":
                            o = th.tensor(obses[i], dtype=th.float32, device=dev).unsqueeze(0)
                            _out = agents[i](o, rnn_hiddens[i])
                            qs = _out[0].squeeze(0)
                            rnn_hiddens[i] = _out[1].detach()
                        else:
                            _out = agents[i](th.tensor(obses[i], dtype=th.float32, device=dev))
                            qs = _out[0] if isinstance(_out, tuple) else _out
                        qs_list.append(qs)

                    eps = schedule.eval(epoch)
                    explore = np.random.random() <= eps
                    actions = []
                    if args.rl_model == "rnn":
                        mac = th.stack(qs_list, dim=0).unsqueeze(0)
                        st = th.tensor(state, dtype=th.float32, device=dev).unsqueeze(0)
                        if explore:
                            actions = [np.random.randint(0, action_spaces[i]) for i in range(n_agent)]
                        else:
                            ja = learner.joint_greedy_actions(mac, st)[0]
                            actions = [int(ja[i].item()) for i in range(n_agent)]
                    else:
                        for i in range(n_agent):
                            qs = qs_list[i]
                            if explore:
                                actions.append(np.random.randint(0, action_spaces[i]))
                            else:
                                actions.append(th.argmax(qs).item())
        
                state, obses, local_rewards, global_reward, done_mask = env.step(actions)
                
                if done_mask == 0:
                    break
                
                transition_data['actions'] = actions
                transition_data['local_rewards'] = local_rewards
                transition_data['global_reward'] = global_reward
                transition_data["states_new"] = obses
                transition_data["global_state_new"] = state
                transition_data["done_mask"] = done_mask
                
                print(global_reward, file=global_reward_file)
                if args.rl_model == "rnn":
                    epoch_episode.append(transition_data)
                else:
                    buffer.add(transition_data)

            if args.rl_model == "rnn" and epoch_episode:
                buffer.add_episode(epoch_episode)
                
            if args.rl_model == "rnn":
                # Sample batch_size subsequences with replacement across stored episodes.
                if len(buffer) > 0:
                    batch = buffer.sample_sequences(args.batch_size, args.seq_len, n_agent)
                    loss, tderror_loss, mono_loss = learner.train(batch)
                    print(loss, file=loss_file)
                    print(tderror_loss, file=tderror_loss_file)
                    print(mono_loss, file=mono_loss_file)
                    print("training epoch:", epoch, "loss:", loss)
            elif len(buffer) >= args.batch_size:
                batch = buffer.sample_batch(args.batch_size)
                loss = learner.train(batch)
                print(loss, file=loss_file)
                print("training epoch:", epoch, "loss:", loss)
        learner.save_models(model_log_dir)
