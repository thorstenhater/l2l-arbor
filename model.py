import arbor as arb
import numpy as np
from random import randrange as rand
import sys
from os.path import abspath as expand

from l2l.utils.experiment import Experiment
from l2l.optimizers.evolution import GeneticAlgorithmOptimizer, GeneticAlgorithmParameters
from l2l.optimizees.arbor.SC import ArbSCOptimizee

fit, swc, ref = list(map(expand, sys.argv[1:]))
name = 'ARBOR-FUN'
results_folder = '../results'
trajectory_name = 'ARBOR'
experiment = Experiment(results_folder)
traj, _ = experiment.prepare_experiment(trajectory_name=trajectory_name,
                                        name=name,
                                        jube_parameter={})
# Innerloop simulator
optimizee = ArbSCOptimizee(traj, fit, swc, ref)
# Outerloop optimizer initialization
parameters = GeneticAlgorithmParameters(seed=0, popsize=10,
                                        CXPB=0.5, MUTPB=0.3, NGEN=10,
                                        indpb=0.02,
                                        tournsize=15,
                                        matepar=0.5,
                                        mutpar=1)
optimizer = GeneticAlgorithmOptimizer(traj,
                                      optimizee_create_individual=optimizee.create_individual,
                                      optimizee_fitness_weights=(-0.1,),
                                      parameters=parameters)
experiment.run_experiment(optimizee=optimizee,
                          optimizer=optimizer,
                          optimizer_parameters=parameters,
                          optimizee_parameters=None)
experiment.end_experiment(optimizer)
