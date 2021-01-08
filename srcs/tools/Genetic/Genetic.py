import datetime
import random


class Genetic:
    mutation: int = 0
    iteration: int = 0
    initial_size: int
    mutation_count: int = 0
    population_size: int
    crossover_count: int = 0
    population: list = []
    past_population: list = []
    end_time: datetime.timedelta
    start_time: datetime.datetime

    def init_population(self):
        """
        create population thanks to population_size
        """
        pass

    def fitness_calculation(self):
        """
        evaluate the accuracy of each individual in population

        @ Need overriding
        """
        pass

    def mating_poll(self):
        """
        Based on the fitness value, the best individuals are selected
        """
        pass

    def parents_selection(self):
        """
        thanks to mating_pool, the parents will be changed
        could add random selection here
        """
        pass

    def _crossover(self, parent1, parent2) -> list:
        """
        take first part of a parent and second part of the other parent to create a child
        """
        self.crossover_count += 1

    def _mutation(self, individual):
        """
        randomly change a value of a gene of a child
        could add other rules than random
        """
        self.mutation_count += 1

    def mating(self):
        """
        mutation: specify the number of mutation in population
        if mutation > population_size, multi mutation can be applied on individual

        apply crossover or mutation
        every operation on individual from the population is here
        """
        pass

    def check_result(self) -> bool:
        """
        check if one individual have the solution
        """
        pass

    def logging(self):
        print(
            f"""
Number of iteration to find solution: {self.iteration}
Number of crossover: {self.crossover_count}
Number of mutation: {self.mutation_count}
Time: {self.end_time}
"""
        )

    def launch(self):
        self.start_time = datetime.datetime.now()
        while self.check_result() is False:
            self.iteration += 1
            self.fitness_calculation()
            self.mating_poll()
            self.parents_selection()
            self.mating()
            print(".", end="")
        self.end_time = datetime.datetime.now() - self.start_time
        self.logging()
