import arbor as arb
import numpy as np
from random import randrange as rand
from dataclasses import dataclass
import logging.config
import os
import yaml
import  sys

from l2l.utils.environment import Environment
from l2l.optimizers.evolution import GeneticAlgorithmOptimizer, GeneticAlgorithmParameters
from l2l.paths import Paths
from l2l.utils import JUBE_runner as jube
from l2l.optimizees.optimizee import Optimizee
from l2l.utils.trajectory import Trajectory

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
        # TODO Fixme
        print(traj)
        # for region, mech, values in traj:
            # cell.paint(f'"{region}"', arb.mechanism(mech, values))
        # TODO End
        cell.place('"center"', arb.iclamp(200, 1000, 0.15))
        model = arb.single_cell_model(cell)
        model.probe('voltage', '"center"', frequency=200000)
        model.properties.catalogue = arb.allen_catalogue()
        model.properties.catalogue.extend(arb.default_catalogue(), '')
        model.run(tfinal=1400, dt=0.005)
        voltages  = np.array(model.traces[0].value[:])
        return (((voltages - self.reference)**2).sum(), )

def main():
    fit, swc, ref = sys.argv[1:]
    name = 'L2L-FUN-GA'
    logger = logging.getLogger(name)
    root_dir_path = "."
    paths = Paths(name, dict(run_no='test'), root_dir_path=root_dir_path)
    traj_file = os.path.join(paths.output_dir_path, 'data.h5')
    # Create an environment that handles running our simulation
    # This initializes an environment
    env = Environment(trajectory=name, filename=traj_file, file_title='{} data'.format(name),
                      comment='{} data'.format(name),
                      add_time=True,
                      automatic_storing=True,
                      log_stdout=False,  # Sends stdout to logs
                      log_folder=os.path.join(paths.output_dir_path, 'logs'))
    # Get the trajectory from the environment
    traj = env.trajectory
    # Set JUBE params
    traj.f_add_parameter_group("JUBE_params", "Contains JUBE parameters")
    # The execution command
    traj.f_add_parameter_to_group("JUBE_params", "exec", "python " + os.path.join(paths.simulation_path, "run_files/run_optimizee.py"))
    # Paths
    traj.f_add_parameter_to_group("JUBE_params", "paths", paths)
    ## Innerloop simulator
    optimizee = ArbSCOptimizee(traj, fit, swc, ref)
    # Prepare optimizee for jube runs
    jube.prepare_optimizee(optimizee, paths.simulation_path)
    ## Outerloop optimizer initialization
    parameters = GeneticAlgorithmParameters(seed=0, popsize=50, CXPB=0.5, MUTPB=0.3, NGEN=100, indpb=0.02, tournsize=15, matepar=0.5, mutpar=1)
    optimizer  = GeneticAlgorithmOptimizer(traj, optimizee_create_individual=optimizee.create_individual, optimizee_fitness_weights=(-0.1,), parameters=parameters)
    # Add post processing
    env.add_postprocessing(optimizer.post_process)
    # Run the simulation with all parameter combinations
    env.run(optimizee.simulate)
    ## Outerloop optimizer end
    optimizer.end()
    # Finally disable logging and close all log-files
    env.disable_logging()

if __name__ == '__main__':
    main()