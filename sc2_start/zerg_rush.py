import sc2
from sc2 import run_game, maps, Race, Difficulty, position
from sc2.player import Bot, Computer
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, \
 CYBERNETICSCORE, STARGATE, VOIDRAY, ZEALOT, STALKER, ROBOTICSFACILITY, \
OBSERVER, IMMORTAL, ADEPT, FORGE, SHIELDBATTERY, PHOTONCANNON, TWILIGHTCOUNCIL, \
DARKSHRINE, DARKTEMPLAR, ORBITALCOMMAND, COMMANDCENTER, DESTRUCTIBLEROCK2X4VERTICAL, \
DESTRUCTIBLEROCK2X4HORIZONTAL, DESTRUCTIBLEROCK2X6VERTICAL, DESTRUCTIBLEROCK2X6HORIZONTAL, \
DESTRUCTIBLEROCK4X4, DESTRUCTIBLEROCK6X6, HATCHERY, ZERGLING, QUEEN, OVERLORD, SPAWNINGPOOL, \
    LARVA, DRONE, AbilityId, EFFECT_INJECTLARVA, ROACH, ROACHWARREN, EXTRACTOR, LAIR, HYDRALISK, HYDRALISKDEN
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
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, LabelBinarizer, QuantileTransformer
import lightgbm as lgb
import traceback
from keras import layers, models, callbacks
import tensorflow
import h5py
import pickle
import time
import pandas as pd

max_iter = 1000000
class_num = 25
var_size = 25
memory_size = 20
nan_replacement = -1
max_game_training_size = 1000
perc_to_consider = .2


lgbm_params =  {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_error',
    "learning_rate": 0.1,
    "max_depth": -1,
    'num_leaves':63
    }

aggressive_units = {
    ZERGLING: [8, 3],
    ROACH: [8, 3],
    HYDRALISK: [8, 3]}
dump_dict = dict()

path = r'C:\Users\trist\Documents\sc2_bot/'
record_memory = 2000


def train_strat_model():
    files = glob.glob(path + '*_data.plk')
    dicts = []

    # if len(files) > max_game_training_size:
    #     files = random.sample(files, max_game_training_size)
    print(files)
    for i in files:
        with open(i, 'rb') as f:
            dicts.append(pickle.load(f))

    random.shuffle(dicts)
    dicts.sort(reverse = True, key = lambda x: x['score'])
    print(len(dicts))
    print([i['score'] for i in dicts])

    # pos_dicts = dicts[:int(len(dicts)*perc_to_consider)]
    # neg_dicts = dicts[-int(len(dicts)*perc_to_consider):]
    pos_dicts = []
    neg_dicts = []
    for i in dicts:
        if i['score'] == 1:
            pos_dicts.append(i)
        else:
            neg_dicts.append(i)

    # print([i['score'] for i in pos_dicts])
    print(len(pos_dicts), len(neg_dicts))
    for i in pos_dicts:
        for j in i['past_moves']:
            j['y'] = 1
            # print(j['game_state'].shape)

    for i in neg_dicts:
        for j in i['past_moves']:
            j['y'] = 0
            # print(j['game_state'].shape)

    full_dicts = pos_dicts + neg_dicts
    features = sum([i['past_moves'] for i in full_dicts], [])

    arrays_x = [i['game_state'] for i in features if max(i['game_state'].shape) == 542]
    x = np.array(arrays_x)
    x2 = np.array([np.array(i['f']) for i in features if max(i['game_state'].shape) == 542])
    y = np.array([i['y'] for i in features if max(i['game_state'].shape) == 542])

    # x2 = np.expand_dims(x2, 1)
    x = np.squeeze(x)
    x = np.hstack([x, x2])
    print(x.shape)

    scaler = StandardScaler()

    scaler.fit(x)
    x = scaler.transform(x)

    with open(path + 'scaler.plk', 'wb') as f:
        pickle.dump(scaler, f)


    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=.1)

    # lgtrain = lgb.Dataset(x_train, y_train)
    # lgvalid = lgb.Dataset(x_val, y_val)
    #
    # model = lgb.train(
    #     lgbm_params,
    #     lgtrain,
    #     num_boost_round=max_iter,
    #     valid_sets=[lgtrain, lgvalid],
    #     valid_names=['train', 'valid'],
    #     early_stopping_rounds=1000,
    #     verbose_eval=10
    # )
    # model.save_model(path + '/lgbmodel', num_iteration=model.best_iteration)


    cb = [callbacks.EarlyStopping(patience=0),
                callbacks.ModelCheckpoint(path + 'dnn.h5',
                                save_best_only=True,
                                save_weights_only=False)]

    model = models.Sequential()
    model.add(layers.Dense(2048, input_dim=567, activation='relu'))
    model.add(layers.Dense(2048, activation='relu'))
    model.add(layers.Dense(2048, activation='relu'))
    model.add(layers.Dense(2048, activation='relu'))

    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=(x_val, y_val), callbacks=cb, epochs=100, batch_size=1024)


def get_closest(unit, unit_list):
    closest_unit = None
    closest_distance = 999

    for i in unit_list:
        if i.distance_to(unit) < closest_distance:
            closest_distance = i.distance_to(unit)
            closest_unit = i
    return closest_unit


def get_distance_of_closest(unit, unit_list):
    closest_unit = None
    closest_distance = 999

    for i in unit_list:
        if i.distance_to(unit) < closest_distance:
            closest_distance = i.distance_to(unit)
            closest_unit = i
    return closest_distance


def get_distance_of_closest_to_hq(pos, unit_list):
    closest_unit = None
    closest_distance = 999

    for i in unit_list:
        if i.distance_to(pos) < closest_distance:
            closest_distance = i.distance_to(pos)
    return closest_distance


def get_closest_distance(units, e_units):
    closest_distance = 999

    for i in units:
        for j in e_units:
            if i.distance_to(j) < closest_distance:
                closest_distance = i.distance_to(j)
    return closest_distance


class Strat():
    random_chance = .1
    print('random', random_chance)
    def __init__(self):
        try:
            self.model = models.load_model(path + 'dnn.h5')
            # self.model = lgb.Booster(model_file=path + '/lgbmodel')
            # with open(path + 'model.plk', 'rb') as f:
            #     self.model = pickle.load(f)

            with open(path + 'scaler.plk', 'rb') as f:
                self.scaler = pickle.load(f)
            # with open(path + 'encoder.plk', 'rb') as f:
            #     self.encoder = pickle.load(f)
            self.use_model = True
        except:
            traceback.print_exc()
            self.use_model = False

    def get_move(self, move_dict, game_state):
        if random.random() < self.random_chance or not self.use_model:
            next_move = random.choice(move_dict)
        else:

            past_moves = [to_one_hot(i) for i in move_dict]
            past_moves = [np.expand_dims(i, 0) for i in past_moves]
            # game_state = np.expand_dims(game_state, 0)

            # print(game_state.shape,  np.array([[0]]).shape)

            xs = np.vstack([np.hstack([game_state, i]) for i in past_moves])
            # xs = [np.reshape(i, (1, -1)) for i in xs]

            # print(xs.shape)
            scaled_input = self.scaler.transform(xs)
            a = self.model.predict(scaled_input)
            # next_move = a[0]
            # print(next_move)
            # p = a[0]
            # a /= a.sum()?

            # # print(p.tolist())
            # next_move_index = np.random.choice(np.array([i for i in range(p.shape[0])]), p = p)
            # next_move_array = [0 for _ in range(p.shape[0])]
            # next_move_array[next_move_index] = 1
            # next_move_array =np.array([next_move_array])
            # next_move = self.encoder.inverse_transform(np.array(next_move_array))[0]
            # print(next_move)
            next_move = np.argmax(a)
            print(next_move)
        return next_move


def to_one_hot(inp, l = var_size):
    out = [0 for _ in range(l)]
    if inp:
        out[inp] = 1
    return np.array(out)


class Partial_RL_Bot(sc2.BotAI):

    def __init__(self, s):
        super().__init__()
        self.ts = int(datetime.datetime.now().timestamp())
        self.iterations_per_counter = 8
        self.iterations_counter = 0
        self.MAX_WORKERS = 100
        self.memory = memory_size
        self.actions = [i for i in range(class_num)]
        self.past_moves = []
        self.s = s
        self.max_score = 0
        self.games_states = []
        self.move_history = []
        self.counter = 0
        self.start_time = time.time()

        for i in range(memory_size + 1):
            self.games_states.append([nan_replacement for j in range(var_size - 1)])
            self.move_history.append(to_one_hot(None, var_size))


    def run_reward_func(self):
        # return len([i for i in self.units if i.name == 'Voidray'])
        # return self.state.score.score + (self.state.score.killed_value_units * 5) - self.minerals - self.vespene
        # return ( self.state.score.score - self.minerals - self.vespene) + (self.state.score.killed_value_units * ( self.state.score.score - self.minerals - self.vespene))
        # return self.state.score.score
        # return self.state.score.killed_value_units + (self.state.score.killed_value_structures * self.state.score.killed_value_structures)
        # self.max_score =  self.state.score.killed_value_units/(max(self.time*self.time, 1))
        # self.max_score = self.state.score.score
        # print(self.state.score.score, max(1, time.time() -  self.start_time))
        self.max_score =  self.state.score.score / (max(1, self.time) * max(1, self.time))


    def get_state(self):
        d1 = self.distance_nexus_to_enemy()
        d2 = self.distance_to_enemy()
        d3 = self.distance_to_enemy_buildings()

        if not d1:
            d1 = -1
        if not d2:
            d2 = -1
        if not d3:
            d3 = -1

        dead_unit_count = len(self.state.dead_units)
        unit_count = len(self.units)
        game_loop = self.state.game_loop
        minerals = self.minerals
        vg = self.vespene
        l_vg = len(self.geysers)
        n_probes = len(self.workers)
        supply_cap = self.supply_cap
        supply_left = self.supply_left
        supply_used = self.supply_used

        num_th = len(self.townhalls)
        keu = len(self.known_enemy_units)
        kes = len(self.known_enemy_structures)

        n_hatch = len([ i for i in self.units if i.name == 'Hatchery'])
        n_drone = len([ i for i in self.units if i.name == 'Drone'])
        n_over = len([ i for i in self.units if i.name == 'Overlord'])
        n_roach = len([ i for i in self.units if i.name == 'Roach'])
        n_hydra = len([ i for i in self.units if i.name == 'Hydralisk'])
        n_queen = len([ i for i in self.units if i.name == 'Queen'])
        n_speed = len([ i for i in self.units if i.name == 'Speedling'])
        n_rw = len([ i for i in self.units if i.name == 'RoachWarren'])
        n_hd = len([ i for i in self.units if i.name == 'HydraliskDen'])
        n_sp = len([ i for i in self.units if i.name == 'SpawningPool'])

        probe_by_nexus = n_probes/max(1, num_th)
        probe_by_units = n_probes/max(1, supply_used)

        ne_hatch = len([ i for i in self.known_enemy_units if i.name == 'Hatchery'])
        ne_drone = len([ i for i in self.known_enemy_units if i.name == 'Drone'])
        ne_over = len([ i for i in self.known_enemy_units if i.name == 'Overlord'])
        ne_roach = len([ i for i in self.known_enemy_units if i.name == 'Roach'])
        ne_hydra = len([ i for i in self.known_enemy_units if i.name == 'Hydralisk'])
        ne_queen = len([ i for i in self.known_enemy_units if i.name == 'Queen'])
        ne_speed = len([ i for i in self.known_enemy_units if i.name == 'Speedling'])
        ne_rw = len([ i for i in self.known_enemy_units if i.name == 'RoachWarren'])
        ne_hd = len([ i for i in self.known_enemy_units if i.name == 'HydraliskDen'])
        ne_sp = len([ i for i in self.known_enemy_units if i.name == 'SpawningPool'])

        n_lar = len([i for i in self.units if i.name == 'Larva'])
        d_to_hq = self.distance_to_enemy_hq()
        # print(n_lar)
        ag_distance = self.ag_unit_distance()

        return [dead_unit_count, unit_count, game_loop, minerals, supply_cap, supply_left, num_th, keu, kes,
                vg, l_vg, n_probes, n_hatch, n_drone, n_over, n_roach, n_hydra, n_queen, n_speed, n_rw, n_hd, n_sp,
                ne_rw, ne_hd, ne_sp,ne_hatch, ne_drone,
                ne_over, ne_roach, ne_hydra, ne_queen, ne_speed, d1, d2, d3, supply_used, probe_by_nexus, probe_by_units,
                self.counter, ag_distance, n_lar, d_to_hq]



    async def on_step(self, iteration):
        global dump_dict

        # self.iterations_per_counter = 10
        self.iterations_counter += 1
        if self.iterations_counter % self.iterations_per_counter == 0:

            times = {}
            self.counter += 1
            self.iteration = iteration
            try:
                # print(self.time)
                if self.time > 720:
                    return
                # self.ex
                self.run_reward_func()

                dump_dict = {'score':self.max_score, 'past_moves':self.past_moves}

                game_state = self.get_state()
                self.games_states.append(game_state)
                t_game_state = self.games_states[-1]
                np_t_game_state = np.expand_dims(np.array(t_game_state), 0)
                np_t_past_moves = np.hstack(self.move_history[-memory_size:])
                np_t_past_moves = np.expand_dims(np_t_past_moves, 0)
                np_t_game_state = np.hstack([np_t_game_state, np_t_past_moves])
                f = self.actions[self.s.get_move(self.actions, np_t_game_state)]

                while ((f in [9, 19, 24]) and random.random() < .8):
                    f = self.actions[self.s.get_move(self.actions, np_t_game_state)]

                self.move_history.append(to_one_hot(f))


                if f == 0:
                    await  self.distribute_workers()
                if f == 1:
                    await  self.build_workers()
                if f == 2:
                    await  self.build_overlord()
                if f == 3:
                    await  self.expand()
                if f == 4:
                    await  self.spawning_pool()
                if f == 5:
                    await  self.build_ling()
                if f == 6:
                    await  self.build_queen()
                if f == 7:
                    await  self.attack()
                if f == 8:
                    await  self.scout()
                if f == 9:
                    await self.retreat()
                if f == 10:
                    await self.defend()
                if f == 11:
                    await self.attack_random_base()
                if f == 12:
                    pass
                    await self.clear_an_obstacle()
                if f == 13:
                    await self.attack_closest_building()
                if f == 14:
                    pass
                    # await self.approach_enemy()
                if f == 15:
                    await self.inject()
                if f == 16:
                    pass
                    await self.build_roach_warren()
                if f == 17:
                    pass
                    await self.build_roach()
                if f == 18:
                    await self.build_extractor()
                if f == 19:
                    await self.retreat_if_close()
                if f == 20:
                    await self.build_lair()
                    pass
                if f == 21:
                    await self.Build_hydra_den()
                    pass
                if f == 22:
                    await self.Build_hydra()
                    pass
                if f == 23:
                    await self.move_random_unit_to_random_loc()
                if f == 24:
                    await self.stop()
                    pass



                # print(np_t_game_state.shape)
                self.past_moves.append({'game_state':np_t_game_state, 'f':to_one_hot(f)})

            except:
                traceback.print_exc()


    async def build_lair(self):
        if self.units(SPAWNINGPOOL).ready.exists:
            if not self.units(LAIR).exists and self.townhalls.first.noqueue:
                if self.can_afford(LAIR):
                    await self.do(self.townhalls.first.build(LAIR))

    async def Build_hydra_den(self):
        if self.units(LAIR).ready.exists:
            if not (self.units(HYDRALISKDEN).exists or self.already_pending(HYDRALISKDEN)):
                if self.can_afford(HYDRALISKDEN):
                    await self.build(HYDRALISKDEN, near=self.townhalls.first.position.towards(self.game_info.map_center, random.randint(2, 8)))



    async def Build_hydra(self):
        while self.can_afford(HYDRALISK) and self.units(LARVA).exists and self.units(HYDRALISKDEN).ready and self.supply_left > 2:
            await self.do(self.units(LARVA).random.train(HYDRALISK))



    async def move_random_unit_to_random_loc(self):
        u = self.units.idle.random
        loc_to_scout = (random.random() * self.game_info.map_size[0], random.random() * self.game_info.map_size[1])
        await self.do(u.move(position.Point2(loc_to_scout)))


    async def build_extractor(self):
        if self.units(EXTRACTOR).amount < 2*len(self.townhalls) and not self.already_pending(EXTRACTOR):
            if self.can_afford(EXTRACTOR):
                drone = self.workers.random
                target = self.state.vespene_geyser.closest_to(drone.position)
                err = await self.do(drone.build(EXTRACTOR, target))


    async def inject(self):
        for queen in self.units(QUEEN):
            # th = self.closest_hatch(queen)
            th = random.choice(self.townhalls)
            abilities = await self.get_available_abilities(queen)
            if AbilityId.EFFECT_INJECTLARVA in abilities:
                await self.do(queen(EFFECT_INJECTLARVA, th))


    async def build_overlord(self):
        if self.can_afford(OVERLORD) and self.units(LARVA).exists:
            await self.do(self.units(LARVA).random.train(OVERLORD))


    async def spawning_pool(self):
        if not (self.units(SPAWNINGPOOL).exists or self.already_pending(SPAWNINGPOOL)):
            if self.can_afford(SPAWNINGPOOL):
                await self.build(SPAWNINGPOOL, near=self.townhalls.first.position.towards(self.game_info.map_center, random.randint(2, 8)))


    async def build_roach_warren(self):
        if not (self.units(ROACHWARREN).exists or self.already_pending(ROACHWARREN)):
            if self.can_afford(ROACHWARREN):
                await self.build(ROACHWARREN, near=self.townhalls.first.position.towards(self.game_info.map_center, random.randint(2, 8)))


    async def build_workers(self):
        while self.can_afford(DRONE) and self.units(LARVA).exists and len([ i for i in self.units if i.name == 'Drone']) < min((22*len(self.townhalls)), 60) and self.supply_left > 1:
            await self.do(self.units(LARVA).random.train(DRONE))


    async def build_ling(self):
        while self.can_afford(ZERGLING) and self.units(LARVA).exists and self.units(SPAWNINGPOOL).ready and self.supply_left > 1:
            await self.do(self.units(LARVA).random.train(ZERGLING))


    async def build_roach(self):
        while self.can_afford(ROACH) and self.units(LARVA).exists and self.units(ROACHWARREN).ready and self.supply_left > 2:
            # break
            await self.do(self.units(LARVA).random.train(ROACH))


    async def build_queen(self):
        th = random.choice(self.townhalls)

        if len([i for i in self.units if i.name == 'Queen']) <= len(self.townhalls)*2:
            if th.is_ready and th.noqueue:
                if self.can_afford(QUEEN):
                    await self.do(th.train(QUEEN))

    async def expand(self):
        if self.can_afford(HATCHERY) and len(self.townhalls):
            await self.expand_now()



    def random_location_variance(self, enemy_start_location):
        x = enemy_start_location[0]
        y = enemy_start_location[1]

        x += ((random.randrange(-40, 40))/100) * enemy_start_location[0]
        y += ((random.randrange(-40, 40))/100) * enemy_start_location[1]

        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x > self.game_info.map_size[0]:
            x = self.game_info.map_size[0]
        if y > self.game_info.map_size[1]:
            y = self.game_info.map_size[1]

        go_to = position.Point2(position.Pointlike((x,y)))
        return go_to


    async def approach_enemy(self):
        for UNIT in aggressive_units:
            for s in self.units(UNIT).idle:
                enemy_location = self.enemy_start_locations[0]
                move_to = self.random_location_variance(enemy_location)
                await self.do(s.move(move_to))


    async def scout(self):
        scout_sent = False
        if len(self.units(OVERLORD)) > 0:
            scout = get_closest(self.enemy_start_locations[0] , self.units(OVERLORD))
            print(scout)
            # scout = self.units(OVERLORD)[0]
            enemy_location = self.enemy_start_locations[0]
            move_to = self.random_location_variance(enemy_location)
            scout_sent = True
            await self.do(scout.move(move_to))

        if len(self.units(DRONE)) > 0 and not scout_sent:
            scout = self.units(DRONE).random
            # scout = random.choice(self.units(PROBE))
            if scout.is_idle:
                enemy_location = self.enemy_start_locations[0]
                move_to = self.random_location_variance(enemy_location)
                scout_sent = True
                await self.do(scout.move(move_to))


        ag_units = []
        for UNIT in aggressive_units:
            for s in self.units(UNIT).idle:
                ag_units.append(s)
        if ag_units and not scout_sent:
            scout = random.choice(ag_units)
            if scout.is_idle:
                enemy_location = self.enemy_start_locations[0]
                move_to = self.random_location_variance(enemy_location)
                await self.do(scout.move(move_to))

    async def find_target(self, unit):
        pass
        # await asyncio.sleep(0.01)

        if len(self.known_enemy_units) > 0:

            return unit.closest_to(self.known_enemy_units)
        elif len(self.known_enemy_structures) > 0:
            return unit.closest_to(self.known_enemy_structures)
        else:
            return self.enemy_start_locations[0]


    async def defend(self):

        for UNIT in aggressive_units:
            for s in self.units(UNIT):
                if len(self.known_enemy_units) > 0  and self.units(NEXUS).ready:
                    if not s.can_attack_air:
                        targets = [i for i in self.known_enemy_units if not i.is_flying]
                    else:
                        targets = [i for i in self.known_enemy_units]
                    target = get_closest(s, targets)
                    if target:
                        await self.do(s.attack(target))


    async def attack(self):
        for UNIT in aggressive_units:
            for s in self.units(UNIT):
                if not s.can_attack_air:
                    targets = [i for i in self.known_enemy_units if not i.is_flying]
                else:
                    targets = [i for i in self.known_enemy_units]
                target = get_closest(s, targets)
                if target:
                    await self.do(s.attack(target))


    async def attack_random_base(self):
        targets = self.known_enemy_structures(ORBITALCOMMAND)

        if targets:
            target = random.choice(targets)
            for UNIT in aggressive_units:
                for s in self.units(UNIT).idle:
                    if target:
                        await self.do(s.attack(target))


    async def attack_closest_building(self):
        targets = self.known_enemy_structures

        if targets:
            for UNIT in aggressive_units:
                for s in self.units(UNIT).idle:
                    target = get_closest(s, self.known_enemy_structures)
                    if target:
                        await self.do(s.attack(target))


    async def clear_an_obstacle(self):
        if self.state.destructables:
            # target = random.choice(self.state.destructables)
            units = []
            for UNIT in aggressive_units:
                for s in self.units(UNIT).idle:
                    units.append(s)
            # units = [i for i in self.units(UNIT).idle for UNIT in aggressive_units]
            if units:
                unit = random.choice(units)
                # for UNIT in aggressive_units:
                #     for s in self.units(UNIT).idle:

                target = get_closest(unit, self.state.destructables)
                if target:
                    await self.do(unit.attack(target))


    async def retreat(self):
        for UNIT in aggressive_units:
            for s in self.units(UNIT).idle:
                target = get_closest(s, self.townhalls)
                if target:
                    await self.do(s.move(target.position))

    async def stop(self):
        for UNIT in aggressive_units:
            for s in self.units(UNIT):
                await self.do(s.stop())


    async def full_retreat(self):
        target = self.townhalls.random

        for UNIT in aggressive_units:
            for s in self.units(UNIT):
                await self.do(s.move(target.position))


    async def attack_own_building(self):

        if self.supply_used > 150:
            targets = [i for i in self.units if i.is_structure]
            target = random.choice(targets)

            for UNIT in aggressive_units:
                for s in self.units(UNIT).idle:
                    await self.do(s.attack(target))


    async def retreat_if_close(self):
        target = self.townhalls.random
        if self.distance_to_enemy() < 5:
            for UNIT in aggressive_units:
                for s in self.units(UNIT):
                    if get_distance_of_closest(s, self.known_enemy_units) < 5:
                        await self.do(s.move(target.position))


    # async def move_to_random_loc(self):
    #     target = (random.random()*, random.random())
    #
    #     for UNIT in aggressive_units:
    #         for s in self.units(UNIT):
    #             target = get_closest(s, self.units(NEXUS))
    #             if target:
    #                 await self.do(s.move(target.position))


    '''Helper functions'''
    def distance_nexus_to_enemy(self):
        return get_closest_distance(self.townhalls, self.known_enemy_units)


    def closest_hatch(self, queen):
        return get_closest(queen, self.townhalls)


    def ag_unit_distance(self):
        ag_units = []
        for UNIT in aggressive_units:
            for s in self.units(UNIT):
                ag_units.append(s)
        return get_closest_distance(ag_units, self.known_enemy_units)


    def distance_to_enemy(self):
        return get_closest_distance(self.units, self.known_enemy_units)


    def distance_to_enemy_buildings(self):
        return get_closest_distance(self.units, self.known_enemy_structures)


    def distance_to_enemy_hq(self):
        return get_closest_distance(self.enemy_start_locations, self.units)


def run_games(d):
    s = Strat()

    ts = int(datetime.datetime.now().timestamp())
    a = None
    print('playing toss at d: ', d )
    start_time = time.time()
    a = run_game(maps.get("AbyssalReefLE"), [
        Bot(Race.Zerg, Partial_RL_Bot(s)),
        Computer(Race.Zerg, d)
        ], realtime=False)

    if a.name == 'Defeat':
        result = 0
        dump_dict['score'] = 0
    elif a.name == 'Tie':
        result = 0
        dump_dict['score'] = 0
    else:
        result = 1
        dump_dict['score'] =  1

    with open('{0}/{1}_data.plk'.format(path, ts), 'wb') as f:
        pickle.dump(dump_dict, f)
    print('game_score:', dump_dict['score'])
    return result, dump_dict['score'], time.time() - start_time

# train_strat_model()

if __name__ == '__main__':
    games = 0.0
    wins = 0.0
    win_rate = 0.0
    difficulties = [Difficulty.VeryEasy, Difficulty.Easy, Difficulty.Medium, Difficulty.Hard]
    record = []

    for i in range(20000):
        print('game num:', i)
        try:
            # train_strat_model()
            if i % 10 == 0 and i >= 0:
                train_strat_model()
        except:
            traceback.print_exc()

        if  win_rate > .8 and i > 6000:
            d = difficulties[3]
        elif  win_rate > .7 and i > 4000:
            d = difficulties[2]
        elif  win_rate > .5 and i > 2000:
            d = difficulties[1]
        else:
            d = difficulties[0]

        # d = difficulties[2]

        result, score, t = run_games(d)
        wins += result
        games += 1

        win_rate = wins / games

        print('win_rate', win_rate, games)
        record.append({'win_rate':win_rate, 'num': i, 'd':d, 'score':score, 'time':t})
        df = pd.DataFrame.from_dict(record)
        df.to_csv('record.csv', index = False)



