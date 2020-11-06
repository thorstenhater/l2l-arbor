import arbor as arb
import numpy as np
from random import randrange as rand
from dataclasses import dataclass
import logging.config
import os
import yaml
import  sys
from collections import defaultdict

from l2l.utils.experiment import Experiment
from l2l.optimizers.evolution import GeneticAlgorithmOptimizer, GeneticAlgorithmParameters
from l2l.utils import JUBE_runner as jube
from l2l.optimizees.optimizee import Optimizee

@dataclass
class PhyPar:
    cm:    float = None
    tempK: float = None
    Vm:    float = None
    rL:    float = None

def load_allen_fit(fit):
    from collections import defaultdict
    import json

    with open(fit) as fd:
        fit = json.load(fd)

    param = defaultdict(PhyPar)
    mechs = defaultdict(dict)
    for block in fit['genome']:
        mech   = block['mechanism'] or 'pas'
        region = block['section']
        name   = block['name']
        value  = float(block['value'])
        if name.endswith('_' + mech):
            name = name[:-(len(mech) + 1)]
        else:
            if mech == "pas":
                # transform names and values
                if name == 'cm':
                    param[region].cm = value/100.0
                elif name == 'Ra':
                    param[region].rL = value
                elif name == 'Vm':
                    param[region].Vm = value
                elif name == 'celsius':
                    param[region].tempK = value + 273.15
                else:
                    raise Exception(f"Unknown key: {name}")
                continue
            else:
                raise Exception(f"Illegal combination {mech} {name}")
        if mech == 'pas':
            mech = 'pas'
        mechs[(region, mech)][name] = value

    param = [(r, vs) for r, vs in param.items()]
    mechs = [(r, m, vs) for (r, m), vs in mechs.items()]

    default = PhyPar(None, # not set in example file
                     float(fit['conditions'][0]['celsius']) + 273.15,
                     float(fit['conditions'][0]['v_init']),
                     float(fit['passive'][0]['ra']))

    erev = []
    for kv in fit['conditions'][0]['erev']:
        region = kv['section']
        for k, v in kv.items():
            if k == 'section':
                continue
            ion = k[1:]
            erev.append((region, ion, float(v)))

    return default, param, erev, mechs

class ArbSCOptimizee(Optimizee):
    def __init__(self, traj, fit, swc, ref):
        self.fns = (fit, swc, ref)

    def create_individual(self):
        fit, swc, ref = self.fns
        import pandas as pd
        segment_tree = arb.load_swc_allen(swc, no_gaps=False)
        self.morphology = arb.morphology(segment_tree)
        self.labels = arb.label_dict({'soma': '(tag 1)', 'axon': '(tag 2)',
                                      'dend': '(tag 3)', 'apic': '(tag 4)',
                                      'center': '(location 0 0.5)'})
        self.reference  = pd.read_csv(ref)['U/mV'].values*1000.0
        self.defaults, self.regions, self.ions, mechs = load_allen_fit(fit)
        # randomise mechanisms
        result = {}
        for r, m, vs in mechs:
            for k, _ in vs.items():
                result[f"{r}.{m}.{k}"] = rand(-1.0, +1.0)
        return result

    def simulate(self, traj):
        cell = arb.cable_cell(self.morphology, self.labels)
        cell.compartments_length(20)
        cell.set_properties(tempK=self.defaults.tempK, Vm=self.defaults.Vm, cm=self.defaults.cm, rL=self.defaults.rL)
        for region, vs in self.regions:
            cell.paint(f'"{region}"', tempK=vs.tempK, Vm=vs.Vm, cm=vs.cm, rL=vs.rL)
        for region, ion, e in self.ions:
            cell.paint(f'"{region}"', ion, rev_pot=e)
        cell.set_ion('ca', int_con=5e-5, ext_con=2.0, method=arb.mechanism('nernst/x=ca'))

        tmp = defaultdict(dict)
        for key, val in traj.individual.items():
            region, mech, valuename = key.split('.')
            tmp[(region, mech)][valuename] = val

        for (region, mech), values in tmp.items():
            cell.paint(f'"{region}"', arb.mechanism(mech, values))

        cell.place('"center"', arb.iclamp(200, 1000, 0.15))
        model = arb.single_cell_model(cell)
        model.probe('voltage', '"center"', frequency=200000)
        model.properties.catalogue = arb.allen_catalogue()
        model.properties.catalogue.extend(arb.default_catalogue(), '')
        model.run(tfinal=1400, dt=0.005)
        voltages = np.array(model.traces[0].value[:])
        return (((voltages - self.reference)**2).sum(), )

def main():
    fit, swc, ref = sys.argv[1:]
    name = 'ARBOR-FUN'
    results_folder = '../results'
    trajectory_name = 'ARBOR'
    experiment = Experiment(results_folder)
    traj, _    = experiment.prepare_experiment(trajectory_name=trajectory_name, name=name, jube_parameter={})
    ## Innerloop simulator
    optimizee = ArbSCOptimizee(traj, fit, swc, ref)
    ## Outerloop optimizer initialization
    parameters = GeneticAlgorithmParameters(seed=0, popsize=50, CXPB=0.5, MUTPB=0.3, NGEN=100, indpb=0.02, tournsize=15, matepar=0.5, mutpar=1)
    optimizer  = GeneticAlgorithmOptimizer(traj, optimizee_create_individual=optimizee.create_individual, optimizee_fitness_weights=(-0.1,), parameters=parameters)
    optimizee  = ArbSCOptimizee(traj, fit, swc, ref)
    parameters = GeneticAlgorithmParameters(seed=0, popsize=50, CXPB=0.5, MUTPB=0.3, NGEN=100, indpb=0.02, tournsize=15, matepar=0.5, mutpar=1)
    optimizer  = GeneticAlgorithmOptimizer(traj, optimizee_create_individual=optimizee.create_individual, optimizee_fitness_weights=(-0.1,), parameters=parameters)
    experiment.run_experiment(optimizee=optimizee, optimizer=optimizer, optimizer_parameters=parameters, optimizee_parameters=None)
    experiment.end_experiment(optimizer)

if __name__ == '__main__':
    main()
