import random
import torch
import gym
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torchtext.vocab import build_vocab_from_iterator
from collections import namedtuple
from models import DQN


TRANSITION = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
alphabets = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
             'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
specials = ['#', '_', '/']
MAX_LEN = 40
CAPACITY = 100000
LR = 1e-3
VAL_RATIO = 0.8
EPSILON = 1e-9
GAMMA = 0.99
BATCH_SIZE = 32
EPOCHS = 100
EPISODES = 100

def get_vocab(alphabets_list=alphabets, specials=specials):
    return build_vocab_from_iterator(alphabets_list, specials=specials)


vocab = get_vocab()


class DataDebug:
    def __init__(self, path='./words_250000_train.txt'):
        with open(path) as f:
            r = f.read().splitlines()
            f.close()
        self.dictionary = r
        self.len = len(self.dictionary)
        print('%d words loaded.' % self.len)

    def __len__(self):
        return self.len

    def split(self, val_ratio=0.2):
        np.random.seed(42)
        all_index = np.random.permutation(self.len)
        n_val = int(self.len * val_ratio)
        n_train = self.len - n_val
        train_index = all_index[:n_train]
        val_index = all_index[n_train:]
        train = [self.dictionary[i] for i in train_index]
        val = [self.dictionary[i] for i in val_index]
        self.train = train
        self.val = val
        return train, val

    def get_max_length(self):
        lengths = [len(i) for i in self.dictionary]
        return max(lengths)


class WordDebug:
    def __init__(self, word):
        # self.word = word.split(' ')
        self.word = word
        self.life = 6
        self.fail = False
        self.all_letters = self.get_letters(self.word)
        self.length = len(word)

    def __len__(self):
        return self.length

    def underscore(self, word):
        underscores = 0
        # word = self.word
        letters = list(set(list(word)))
        while underscores < len(word) * 0.5:
            chosen = random.choice(letters)
            underscores += word.count(chosen)
            letters.remove(chosen)
            word = word.replace(chosen, '_')
        return word

    # useful
    def get_letters(self, word):
        word.replace('_', '')
        return list(set(list(word)))

    def padding(self, word, attempts=''):
        padding_size = MAX_LEN - len(word) - 1 - len(attempts)
        word = attempts + '/' + word + '#' * padding_size
        return word

    def action(self, guess_letter):
        assert guess_letter not in self.letters

        if guess_letter in self.all_letters:
            for i in range(self.length):
                if self.word[i] == guess_letter:
                    assert self.working[i] == '_'
                    self.working[i] = guess_letter
            self.letters.append(guess_letter)
        else:
            self.life -= 1
            if self.life <= 0:
                self.fail = True
            if '/' not in self.word:
                self.word += '/'
            self.word += guess_letter


def get_underscore(word):
    underscores = 0
    # word = self.word
    letters = list(set(list(word)))
    while underscores < len(word) * 0.5:
        chosen = random.choice(letters)
        underscores += word.count(chosen)
        letters.remove(chosen)
        word = word.replace(chosen, '_')
    return word


def get_padding(word, attempts=''):
    padding_size = MAX_LEN - len(word) - 1 - len(attempts)
    word = attempts + '/' + word + '#' * padding_size
    return word


def get_word(token_tensor):
    l = vocab.lookup_tokens(list(token_tensor))
    s = ''.join(l)
    return s.split('/')[-1].replace('#', '')


def get_depadding(word):
    return word.split('/')[-1].replace('#', '')


class HangmanDataset(Dataset):
    def __init__(self, dataset):
        self.raw = dataset
        self.vocab = get_vocab()
        self.uds = [get_underscore(word) for word in self.raw]
        self.inputs = [vocab(list(get_padding(word))) for word in self.uds]
        self.template = [vocab(list(get_padding(word))) for word in self.raw]
        self.inputs = torch.tensor(self.inputs)
        self.template = torch.tensor(self.template)
        if torch.cuda.is_available():
            self.inputs.to('cuda')
            self.template.to('cuda')

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, idx):
        inputs = self.inputs[idx]
        template = self.template[idx]
        raw = self.raw[idx]
        return inputs, template, raw


class Environment(gym.Env):
    def __init__(self, split_data):
        self.split_data = split_data
        super().__init__()
        self.reset()

    # def step(self, action):
    #     # action: str(letter)
    #     reward = 0
    #     if self.life < 0:
    #         self.stop = True
    #         reward -= 1
    #
    #     else:
    #         if get_word(self.state.cpu()) == self.truth:
    #             self.stop = True
    #             reward += 1
    #         else:
    #             if action in self.truth:
    #                 reward += 0.1
    #             else:
    #                 reward -= 0.1
    #                 self.life -= 1
    #             self.update(action)
    #
    #     self.attempts += action
    #     return {"state": self.state, "action": action, "reward": reward, "attempts": self.attempts,
    #             "life": self.life, 'stop': self.stop}

    def step(self, action):
        # action: str(letter)
        reward = 0
        if action in self.truth and action not in self.word:
            reward += 0.1
        else:
            reward -= 0.1
            self.life -= 1
        self.update(action)

        if self.life <= 0:
            self.stop = True
            reward -= 1
        else:
            if get_word(self.state.cpu()) == self.truth:
                self.stop = True
                reward += 1

        self.attempts += action
        return {"state": self.state, "action": action, "reward": reward, "attempts": self.attempts,
                "life": self.life, 'stop': self.stop}

    def reset(self):
        self.train_set = HangmanDataset(self.split_data)
        data = self.train_set.__getitem__(random.randint(0, len(self.train_set)))
        # data = self.train_set.__getitem__(0) # debug
        self.truth = data[2]
        self.template = data[1]
        self.state = data[0]
        self.word = ''.join(vocab.lookup_tokens(list(self.state.cpu())))
        self.life = 6
        self.attempts = ''
        self.stop = False

    def update(self, action):
        token = vocab([action])[0]
        self.word = ''.join(vocab.lookup_tokens(list(self.state.cpu())))
        if action in self.truth and action not in self.word:
            mask = self.template == token
            self.state += (mask * (token - 1))
        else:
            self.state = torch.cat([torch.tensor([token]).to(self.state.device), self.state[:-1]])
            self.template = torch.cat([torch.tensor([token]).to(self.template.device), self.template[:-1]])

    def action_pool(self):
        if not hasattr(self, 'actions'):
            self.actions = torch.tensor(vocab(alphabets))
        return self.actions


class Agent:
    def __init__(self, tr, vl):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.memory = []
        self.position = 0
        self.steps_done = 0
        self.env = Environment(tr)
        self.val_env = Environment(vl)

        self.policy_net, self.target_net = DQN(), DQN()

        self.update_count = 0
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR)
        self.loss_func = torch.nn.MSELoss()
        self.train_reward_list = []

    def store_transition(self, transition):
        if len(self.memory) < CAPACITY:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % CAPACITY

    def select_action(self, state):
        value = self.policy_net(state, self.env.action_pool())
        action_max_value, index = torch.max(value, 1)
        action = alphabets[index.item()]
        if np.random.rand(1) >= 0.9:  # epslion greedy
            action = alphabets[np.random.choice(range(0, 26), 1).item()]
        return action

    def update(self):
        state = torch.cat([t.state.view(1, -1) for t in self.memory], dim=0)
        action = torch.tensor(vocab([t.action for t in self.memory]))
        reward = torch.tensor([t.reward for t in self.memory])
        next_state = torch.cat([t.next_state.view(1, -1) for t in self.memory], dim=0)
        losses = []
        # print(state, action, reward, next_state)

        reward = (reward - reward.mean()) / (reward.std() + EPSILON)
        with torch.no_grad():
            target_v = reward + GAMMA * self.target_net(next_state, self.env.action_pool()).max(1)[0]
        for _ in range(10):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.memory))), batch_size=BATCH_SIZE, drop_last=False):
                # v = (self.policy_net(state).gather(1, action))[index]
                loss = self.loss_func(target_v[index].unsqueeze(1), (self.policy_net(state, action))[index])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # self.writer.add_scalar('loss/value_loss', loss, self.update_count)
                losses.append(loss.item())
                self.update_count += 1
                if self.update_count % 100 == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
        return sum(losses) / len(losses)

def train():
    tr, vl = DataDebug().split(VAL_RATIO)
    for epoch in range(EPOCHS):
        agent = Agent(tr, vl)
        for i_ep in range(EPISODES):
            agent.env.reset()
            while True:
                state = agent.env.state
                action = agent.select_action(state)
                result = agent.env.step(action)
                # print(result)
                next_state = result['state']
                reward = result['reward']
                stop = result['stop']
                transition = TRANSITION(state, action, next_state, reward)
                agent.store_transition(transition)
                agent.env.state = next_state
                if stop:
                    break

        epoch_loss = agent.update()
        print('Epoch %d: Loss = %.4f' % (epoch, epoch_loss))


train()