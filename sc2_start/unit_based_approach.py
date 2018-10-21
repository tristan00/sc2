import sc2
from sc2 import run_game, maps, Race, Difficulty, position
from sc2.player import Bot, Computer
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, \
 CYBERNETICSCORE, STARGATE, VOIDRAY, ZEALOT, STALKER, ROBOTICSFACILITY, \
OBSERVER, IMMORTAL, ADEPT, FORGE, SHIELDBATTERY, PHOTONCANNON, TWILIGHTCOUNCIL, \
DARKSHRINE, DARKTEMPLAR, ORBITALCOMMAND, COMMANDCENTER, DESTRUCTIBLEROCK2X4VERTICAL, \
DESTRUCTIBLEROCK2X4HORIZONTAL, DESTRUCTIBLEROCK2X6VERTICAL, DESTRUCTIBLEROCK2X6HORIZONTAL, \
DESTRUCTIBLEROCK4X4, DESTRUCTIBLEROCK6X6
import multiprocessing
import random
import asyncio
import numpy as np
import pickle
import datetime
import glob
import operator
import os
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, LabelBinarizer
import lightgbm as lgb
import traceback
from keras import layers, models, callbacks
import tensorflow
import h5py
import pickle
import time
import pandas as pd



max_iter = 1000000
class_num = 30
var_size = 41
memory_size = 500
nan_replacement = -1
max_game_training_size = 1000
perc_to_consider = .5


aggressive_units = {
    ZEALOT: [8, 3],
    STALKER: [8, 3],
    IMMORTAL: [8, 3],
    VOIDRAY: [8, 3],
    ADEPT: [8, 3],
    DARKTEMPLAR: [8, 3],
}

buildings = {NEXUS,PYLON,ASSIMILATOR,GATEWAY,CYBERNETICSCORE,STARGATE,ROBOTICSFACILITY,TWILIGHTCOUNCIL,DARKSHRINE}

dump_dict = dict()

path = r'C:\Users\trist\Documents\sc2_bot/'
record_memory = 2000


def train_strat_model():
    files = glob.glob(path + '*_data.plk')
    dicts = []

    # if len(files) > max_game_training_size:
    #     files = random.sample(files, max_game_training_size)

    for i in files:
        with open(i, 'rb') as f:
            dicts.append(pickle.load(f))

    random.shuffle(dicts)
    dicts.sort(reverse = True, key = lambda x: x['score'])
    print(len(dicts))
    print([i['score'] for i in dicts])

    features = sum([i['past_moves'] for i in dicts], [])
    # random.shuffle(features)
    x = np.array([i['game_state'] for i in features])
    y = np.array([i['f'] for i in features])

    # x = x.reshape((-1, var_size * memory_size))
    x = np.squeeze(x)
    scaler = MinMaxScaler()
    scaler.fit(x)
    enc = LabelBinarizer(sparse_output = False)
    enc.fit(np.reshape(y, (-1, 1)))

    with open(path + 'scaler.plk', 'wb') as f:
        pickle.dump(scaler, f)
    with open(path + 'encoder.plk', 'wb') as f:
        pickle.dump(enc, f)

    pos_dicts = dicts[:int(len(dicts)*perc_to_consider)]
    neg_dicts = dicts[-int(len(dicts)*perc_to_consider):]
    pos_features = sum([i['past_moves'] for i in pos_dicts], [])
    neg_features = sum([i['past_moves'] for i in neg_dicts], [])

    pos_x = np.array([i['game_state'] for i in pos_features])
    neg_x = np.array([i['game_state'] for i in neg_features])
    pos_y = np.array([i['f'] for i in pos_features])
    neg_y = np.array([i['f'] for i in neg_features])

    # pos_x = pos_x.reshape((-1, var_size * memory_size))
    # neg_x = neg_x.reshape((-1, var_size * memory_size))
    pos_x = np.squeeze(pos_x)
    neg_x = np.squeeze(neg_x)

    pos_x = scaler.transform(pos_x)
    neg_x = scaler.transform(neg_x)

    pos_y = enc.transform(np.reshape(pos_y, (-1, 1)))
    neg_y = enc.transform(np.reshape(neg_y, (-1, 1)))

    x = np.vstack([pos_x, neg_x])
    pos_y = pos_y.astype(np.float32)
    neg_y = (neg_y == 0).astype(np.float32)
    y = np.vstack([pos_y, neg_y])

    # model = ExtraTreesClassifier(n_jobs=-1, min_samples_leaf = 50)
    # model.fit(pos_x, pos_y)
    #
    # with open(path + 'model.plk', 'wb') as f:
    #     pickle.dump(model, f)
    #
    # res = model.predict(x)
    # print(res)
    #


    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=.1)
    #
    cb = [callbacks.EarlyStopping(patience=0),
                callbacks.ModelCheckpoint(path + 'dnn.h5',
                                save_best_only=True,
                                save_weights_only=False)]

    model = models.Sequential()
    model.add(layers.Dense(1000, input_dim=40 + memory_size, activation='elu'))
    model.add(layers.Dense(1000, activation='elu'))
    model.add(layers.Dense(1000, activation='elu'))
    model.add(layers.Dense(class_num, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mae'])
    model.fit(x_train, y_train, validation_data=(x_val, y_val), callbacks=cb, epochs=100, batch_size=512)


def get_closest(unit, unit_list):
    closest_unit = None
    closest_distance = 999

    for i in unit_list:
        if i.distance_to(unit) < closest_distance:
            closest_distance = i.distance_to(unit)
            closest_unit = i
    return closest_unit


def get_closest_distance(units, e_units):
    closest_distance = 999

    for i in units:
        for j in e_units:
            if i.distance_to(j) < closest_distance:
                closest_distance = i.distance_to(j)
    return closest_distance


class Strat():
    random_chance = .01
    print('random', random_chance)
    def __init__(self):
        try:
            self.model = models.load_model(path + 'dnn.h5')
            # with open(path + 'model.plk', 'rb') as f:
            #     self.model = pickle.load(f)

            with open(path + 'scaler.plk', 'rb') as f:
                self.scaler = pickle.load(f)
            with open(path + 'encoder.plk', 'rb') as f:
                self.encoder = pickle.load(f)
            self.use_model = True
        except:
            traceback.print_exc()
            self.use_model = False

    def get_move(self, move_dict, game_state):
        if random.random() < self.random_chance or not self.use_model:
            next_move = random.choice(move_dict)
        else:
            game_state = np.expand_dims(game_state, 0)
            # print(game_state.shape)
            game_state = np.reshape(game_state, (1, -1))
            # print(game_state.shape)

            # game_state = game_state.reshape((-1, var_size * memory_size))
            scaled_input = self.scaler.transform(game_state)
            a = self.model.predict(scaled_input)
            # next_move = a[0]
            # print(next_move)
            p = a[0]
            p /= p.sum()

            # print(p.tolist())
            next_move_index = np.random.choice(np.array([i for i in range(p.shape[0])]), p = p)
            next_move_array = [0 for _ in range(p.shape[0])]
            next_move_array[next_move_index] = 1
            next_move_array =np.array([next_move_array])
            next_move = self.encoder.inverse_transform(np.array(next_move_array))[0]
            # print(next_move)
            # next_move = np.argmax(a[0])
        return next_move


class UnitBot(sc2.BotAI):

    def __init__(self, s):
        super().__init__()
        self.ts = int(datetime.datetime.now().timestamp())
        self.ITERATIONS_PER_MINUTE = 60
        self.MAX_WORKERS = 100
        self.memory = memory_size
        self.actions = [i for i in range(class_num)]
        self.past_moves = []
        self.s = s
        self.max_score = 0
        self.games_states = []
        self.move_history = []
        self.counter = 0

        for i in range(memory_size + 1):
            self.games_states.append([nan_replacement for j in range(var_size - 1)])
            self.move_history.append(-1)


    def get_possible_moves(self):
        possible_moves = []

        for i in self.units:
            possible_moves.append({'move_id':0, 'unit':i, 'x':random.random(), 'y':random.random()})

        for i in self.workers:
            for b in buildings:
                possible_moves.append({'move_id':1, 'unit':i, 'building_id': b, 'x':random.random(), 'y':random.random()})



    def move(self, unit, v1, v2):
        unit.move((v1 * self.game_info.map_size[0], v2 * self.game_info.map_size[1]))


    def build_building(self, u, b, v1, v2):
        self.build(building=b, unit=u, near=position.Point2((v1 * self.game_info.map_size[0], v2 * self.game_info.map_size[1])))


    def get_closest_enemy_to_pos(self, x, y):
        pass


    def read_map(self):
        enemy_df = []
        enemy_unit_dict = {}
        for i in self.known_enemy_units:
            enemy_unit_dict[i.tag] = i
            enemy_df.append({'tag':i.tag, 'x':i.position[0], 'y':i.position[1]})
        self.enemy_df = pd.DataFrame.from_dict(enemy_df)

        unit_df = []
        unit_dict = {}
        for i in self.units:
            if i.name.upper() in aggressive_units.keys():
                unit_dict[i.tag] = i
                unit_df.append({'tag':i.tag, 'x':i.position[0], 'y':i.position[1]})
        self.unit_df = pd.DataFrame.from_dict(unit_df)





def run_games():
    s = Strat()

    ts = int(datetime.datetime.now().timestamp())
    a = None
    print('playing easy')
    a = run_game(maps.get("AbyssalReefLE"), [
        Bot(Race.Protoss, UnitBot(s)),
        Computer(Race.Terran, Difficulty.Easy)
        ], realtime=False)

    with open('{0}/{1}_data.plk'.format(path, ts), 'wb') as f:
        pickle.dump(dump_dict, f)

if __name__ == '__main__':
    games = 0
    wins = 0
    win_rate = 0
    difficulties = [0]
    for i in range(20000):

        try:
            if i % 1 == 0 and i != 0:
                train_strat_model()
        except:
            traceback.print_exc()
        run_games()
        # pool = [multiprocessing.Process(target=run_games) for i in range(6)]
        # for p in pool:
        #     p.start()
        #     time.sleep(5)
        # [p.join() for p in pool]


