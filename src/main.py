"""Ludo AI project.

This module contains the main script for the Ludo AI project.

@Author: Emanuel-Ionut Otel
@Company: University of Southern Denmark
@Created: 2022-06-15
@Contact: emote21@student.sdu.dk
"""

#### ---- IMPORTS AREA ---- ####
import os,csv, ludopy, torch
import numpy as np

from tqdm import tqdm
from deep_q_net import Agent
from typing import List, Tuple, Union 
#### ---- IMPORTS AREA ---- ####

#### ---- GLOBAL INIT AREA ---- ####
f = open(os.getcwd() + '/game_stats_random.csv', 'w')
writer = csv.writer(f)
header = ['number_of_game', 
          'winner', 
          'cur_reward', 
          'avg100_reward', 
          'avgall_reward', 
          'epsilon', 
          'win3', 
          'win2', 
          'win1',
          'avg3', 
          'avg2', 
          'avg1']
torch.cuda.empty_cache()
g = ludopy.Game()
agent = Agent(gamma=0.10, epsilon=1, batch_size=int(3e04), eps_end=1e-2, input_dims=17, lr=1e-2)
scores, eps_hist = [], []
avg_score = 0
last_score = 1.0
gama = 0.5
#### ---- GLOBAL INIT AREA ---- ####


def random_move(move_pieces: np.ndarray) -> Union[List, int]:
    """This function randomly chooses a piece to move.
    
    :param move_pieces: List of pieces that can be moved

    :return: The piece to move
    """
    if len(move_pieces):
        piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
    else:
        piece_to_move = -1
    return piece_to_move


def get_state(observations: Tuple[int, np.ndarray, np.ndarray, np.ndarray, bool, bool]) -> np.ndarray:
    """This function returns the state of the game, based on the observations.

    :param observations: The observations of the game

    :return: The state of the game
    """
    (dice, _, player_pieces, enemy_pieces, _, _) = observations
    index = 1
    state = np.zeros(17)
    state[index] = dice
    for i in player_pieces:
        state[index] = i
        index += 1
    for i in enemy_pieces:
        for j in i:
            state[index] = j
            index += 1
    return state


def return_reward(old_observations:Tuple, new_observations:Tuple) -> int:
    """This function returns the reward of the game, based on the observations.
    
    :param old_observations: The observations of the past state, before making a decition
    :param new_observations: The observations of the current state, after making a decition
    
    :return: The reward of the game"""
    reward = 0
    (_, _, old_player_pieces, old_enemy_pieces, _, _) = old_observations[0]
    (_, _, new_player_pieces, new_enemy_pieces, player_is_a_winner, _) = new_observations
    if player_is_a_winner:
        reward += 6000
    for i in range(4):
        if old_player_pieces[i] == 0 and new_player_pieces[i] != 0:
            reward += 50
        if old_player_pieces[i] != 0 and new_player_pieces[i] == 0:
            reward -= 500
        if old_player_pieces[i] <= 53 and new_player_pieces[i] >= 54:
            reward += 100
        if old_player_pieces[i] != 59 and new_player_pieces[i] == 59:
            reward += 200
        for j in range(3):
            if old_player_pieces[i] != old_enemy_pieces[j][i] and old_enemy_pieces[j][i] != 0 and new_player_pieces[i] == old_enemy_pieces[j][i] and new_enemy_pieces[j][i] == 0:
                reward += 500
    return reward


def get_action_space(move_pieces: np.ndarray) -> List:
    """This function returns the action space of the game, based on the observations.

    :param move_pieces: The pieces that can be moved

    :return: The action space of the game
    """
    action_space = [0, 0, 0, 0]
    for i in move_pieces:
        action_space[i] = 1
    return action_space


def one_game(learn:bool=True, players_blocked:int=0) -> Tuple[int, float, float, float]:
    """This function plays one game of Ludo.

    :param learn: If the agent should learn from the game
    :param players_blocked: The number of players that should be blocked

    :return: The winner, the score, the average score of the last 100 games, the average score of all games
    """
    g.reset()
    block_player=[]
    if players_blocked != 0:
        for i in range(1, players_blocked):
            block_player.append(i)
        g.ghost_players=block_player
    score = 0
    there_is_a_winner = False
    while not there_is_a_winner:
        old_observations = g.get_observation()
        player_i = old_observations[1]
        (_, move_pieces, _, _, _, there_is_a_winner) = old_observations[0]
        reward = 0
        if player_i != 0:
            piece_to_move = random_move(move_pieces)
            _, _, _, _, _, there_is_a_winner = g.answer_observation(piece_to_move)
        else:
            with torch.no_grad():
                piece_to_move = agent.choose_action(get_state(old_observations[0]), get_action_space(move_pieces))
            new_observations = g.answer_observation(piece_to_move)
            reward += return_reward(old_observations, new_observations)
            score += reward
            _, move_pieces_, _, _, _, there_is_a_winner = new_observations
            agent.store_transition(state=get_state(old_observations[0]), action=piece_to_move, reward=reward,
                                   state_=get_state(new_observations), action_space=get_action_space(move_pieces),
                                   action_space_=get_action_space(move_pieces_), done=there_is_a_winner)
    if learn:
        agent.Q_eval.train()
        agent.learn()
        agent.Q_eval.eval()

    global last_score
    final_score = (score + gama * last_score) / 2
    last_score = final_score
    scores.append(score)
    eps_hist.append(agent.epsilon)
    avg_score_now = np.mean(scores[-100:])
    all_time_avg = np.mean(scores[:])

    return g.game_winners[0], score, avg_score_now, all_time_avg


def test_games(players_blocked:int=0) -> Tuple[int, float]:
    """This function plays 1 game of Ludo, without any training going on.

    :param players_blocked: The number of players that should be blocked

    :return: The average score of the game plus the winner.
    """
    g.reset()
    block_player=[]
    if players_blocked != 0:
        for i in range(1, players_blocked):
            block_player.append(i)
        g.ghost_players=block_player
    score = 0
    there_is_a_winner = False
    while not there_is_a_winner:
        old_observations = g.get_observation()
        player_i = old_observations[1]
        (_, move_pieces, _, _, _, there_is_a_winner) = old_observations[0]
        reward = 0
        if player_i != 0:
            piece_to_move = random_move(move_pieces)
            _, _, _, _, _, there_is_a_winner = g.answer_observation(piece_to_move)
        else:
            with torch.no_grad():
                piece_to_move = agent.choose_action(get_state(old_observations[0]), get_action_space(move_pieces))
            new_observations = g.answer_observation(piece_to_move)
            reward += return_reward(old_observations, new_observations)
            score += reward
    return g.game_winners[0], score


def learn_games(number_of_game:int=100, learn:bool=True, players_blocked:int=0) -> List[int]:
    """This function plays a number of games of Ludo.

    :param number_of_game: The number of games that should be played
    :param learn: If the agent should learn from the games
    :param players_blocked: The number of players that should be blocked

    :return: A list with the number of wins of each player
    """
    wins = [0, 0, 0, 0]
    writer.writerow(header)
    for i in tqdm(range(0, number_of_game), desc="Current game"):
        game_winner, score, avg_score_cur, all_avg = one_game(learn, players_blocked=players_blocked)
        winrates = [0 ,0 ,0 ,0]
        avg_rewards = [0 ,0 ,0 ,0]
        if i%100==0:
            winrates, avg_rewards = test()
        wins[game_winner] += 1
        writer.writerow([str(i), str(game_winner), str(score), str(avg_score_cur), str(all_avg), str(agent.epsilon), 
                         str(winrates[0]), str(winrates[1]), str(winrates[2]), str(avg_rewards[0]), str(avg_rewards[1]), str(avg_rewards[2])])
        tqdm.write('Episode: ' + str(i) +
                   ' Score: ' + str(round(score, 2)) +
                   ' Avg_score: ' + str(round(avg_score_cur, 2)) +
                   ' All_Avg: ' + str(round(all_avg, 2)) +
                   ' Epsilon: ' + str(round(agent.epsilon, 2))+
                   ' Win3: ' + str(round(winrates[0], 2))+
                   ' Win2: ' + str(round(winrates[1], 2))+
                   ' Win1: ' + str(round(winrates[2], 2))+
                   ' Avg3: ' + str(round(avg_rewards[0], 2))+
                   ' Avg2: ' + str(round(avg_rewards[1], 2))+
                   ' Avg1: ' + str(round(avg_rewards[2], 2)))
    return wins


def test(number_of_game:int=100):
    avg_rewards = [0., 0, 0.]
    winrates = [0., 0., 0.]
    for j in range(3):
        winner = [0, 0, 0, 0]
        rewards = []
        for _ in range(number_of_game):
            game_winner, reward = test_games(players_blocked=j)
            winner[game_winner] += 1
            rewards.append(reward)
        avg_rewards[j] = np.mean(rewards)
        winrates[j] = (winner[0]/number_of_game)*100
    return winrates, avg_rewards


if __name__ == '__main__':
    agent.Q_eval.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    number_of_games = int(6e3)
    stats = learn_games(number_of_games, players_blocked=0)
    agent.save()
    print('Teaching Win-rate: ' + str(stats[0] / number_of_games * 100) + '%')
    number_of_games = int(5e2)
    agent.epsilon = 0
    agent.Q_eval.eval()
    winrates, rewards = test(number_of_game=number_of_games)
    print('True Win-rate: ' + str(winrates[0]) + '%')

f.close()
