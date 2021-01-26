import datetime
import logging
import threading

from numpy import random
import numpy as np

from tools.Network import Network
from tools.Genetic.Genetic import Genetic


class PerceptronGenetic(Genetic):
    best_ind = Network

    def _create_layers(self):
        layer = []
        for _ in range(random.randint(2, 6)):
            layer.append(random.randint(self.output_size, self.input_size))
        layer = np.sort(layer)[::-1]
        return np.array([*layer, self.output_size])

    def init_population(self):
        self.population = []
        for i in range(self.population_size):
            epochs = random.randint(300, 1000)
            learning_rate = random.uniform(0.001, 0.1)
            layers_size = self._create_layers()
            self.population.append(
                {
                    "obj": Network(
                        input_dim=30,
                        layers_size=layers_size,
                        epochs=epochs,
                        learning_rate=learning_rate,
                    ),
                    "grade": 0,
                }
            )
            self.population[-1]["obj"].name = (f"thread-{i}",)

    def _train_network(self, nn):
        nn.train()

    def _launch_obj(self):
        threads = []
        for i, individual in enumerate(self.population):
            individual["obj"].name = f"thread {i}"
            x = threading.Thread(target=self._train_network, args=[individual["obj"]])
            x.start()
            threads.append(x)
        for x in threads:
            x.join()

    def fitness_calculation(self):
        self._launch_obj()
        self.best_ind = sorted(
            self.population, key=lambda ind: ind["obj"].best_loss[0]
        )[0]["obj"]
        logging.warning(
            f"""best loss: {self.best_ind.best_loss[0]}
        layers = {self.best_ind.layers_size} | epochs = {self.best_ind.epochs} | learning_rate = {self.best_ind.learning_rate}
        """,
        )
        for i, ind in enumerate(self.population):
            self.population[i]["grade"] += ind["obj"].best_loss[0] * 200

    def mating_poll(self):
        """
        select the first half of the population with the highest match
        """
        self.past_population = self.population
        self.population = sorted(
            self.population, key=lambda ind: ind["grade"], reverse=True
        )[: int(self.population_size / 2)]

    def parents_selection(self):
        if len(self.population) < self.initial_population:
            self.population.extend(
                [
                    self.past_population[random.randint(0, self.population_size - 1)]
                    for _ in range(self.population_size - len(self.population))
                ]
            )
        random.shuffle(self.population)
        self.population_size = len(self.population)

    def _random_crossover(self, parent1, parent2):
        args = {
            "input_dim": self.input_size,
            "layers_size": parent1.layers_size,
            "epochs": parent1.epochs,
            "learning_rate": parent1.learning_rate,
        }
        for k in ["layers_size", "epochs", "learning_rate"]:
            if (random.randint(0, 9) % 2) == 1:
                args[k] = vars(parent2)[k]
        return args

    def _crossover(self, parent1, parent2) -> list:
        super()._crossover(parent1, parent2)
        args1 = self._random_crossover(parent1["obj"], parent2["obj"])
        args2 = self._random_crossover(parent1["obj"], parent2["obj"])
        return [
            {"obj": Network(**args1), "grade": 0},
            {"obj": Network(**args2), "grade": 0},
        ]

    def _random_mutation(self, individual, max_mut):
        mut = 0
        args = {
            "input_dim": self.input_size,
            "layers_size": individual.layers_size,
            "epochs": individual.epochs,
            "learning_rate": individual.learning_rate,
        }
        for k in ["layers_size", "epochs", "learning_rate"]:
            if random.randn() > 0.3 and mut < max_mut:
                mut += 1
                if k == "layers_size":
                    args[k] = self._create_layers()
                elif k == "epochs":
                    args[k] = random.randint(1000, 3000)
                elif k == "learning_rate":
                    args[k] = random.uniform(0.001, 0.1)
        return args

    def _mutation(self, individual):
        super()._mutation(individual)
        args = self._random_mutation(individual["obj"], 1)
        individual["obj"] = Network(**args)
        individual["grade"] = 0

    def check_result(self) -> bool:
        timer = datetime.timedelta(minutes=self.timer)
        if (
            datetime.datetime.now() - self.start_time < timer
            or self.best_ind["obj"].best_loss[0] > self.loss
        ):
            return False
        self.best_ind["obj"].save_model(model_name="genetic_model_0_07")
        return True

    def mating(self):
        temp_pop = []
        for _ in range(int(self.population_size / 2)):
            parent1 = random.randint(0, len(self.population) - 1)
            parent2 = random.randint(0, len(self.population) - 1)
            temp_pop.extend(
                self._crossover(
                    self.population.pop(parent1), self.population.pop(parent2)
                )
            )
        if self.mutation > 0:
            for _ in range(self.mutation):
                self._mutation(temp_pop[random.randint(0, self.population_size - 1)])
        self.population = temp_pop
        self.population_size = len(self.population)

    def __init__(self, population_size, mutation, timer, loss, input_size, output_size):
        self.loss = loss
        self.timer = timer
        self.mutation = mutation
        self.population_size = population_size
        self.initial_population = population_size
        self.input_size = input_size
        self.output_size = output_size
        self.init_population()
