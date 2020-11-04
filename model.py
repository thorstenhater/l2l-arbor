import arbor as arb
import numpy as np
import pandas as pd
from random import randrange as rand
from dataclasses import dataclass

from l2l.optimizees.optimizee import Optimizee
from l2l.utils.trajectory import Trajectory

@dataclass
class parameters:
    cm:    float = None
    tempK: float = None
    Vm:    float = None
    rL:    float = None

def load_allen_fit(fit):
    from collections import defaultdict
    import json

    with open(fit) as fd:
        fit = json.load(fd)

    param = defaultdict(parameters)
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

    default = parameters(None, # not set in example file
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
    def __init__(self, traj):
        self.segment_tree = arb.load_swc_allen('cell.swc', no_gaps=False)
        self.morphology = arb.morphology(segment_tree)
        self.labels = arb.label_dict({'soma': '(tag 1)', 'axon': '(tag 2)',
                                      'dend': '(tag 3)', 'apic': '(tag 4)',
                                      'center': '(location 0 0.5)'})
        self.reference  = pd.read_csv('nrn.csv')['U/mv'].values()*1000.0
        params = load_allen_fit('fit.json')
        self.defaults, self.regions, self.ions, self.mechanisms = params

        # randomise mechanisms
        for idx in len(self.mechanisms):
            r, m, vs = self.mechanisms[idx]
            vs = { k: rand(-1.0, +1.0) for k, v in vs.items() }
            self.mechanisms[idx] = (r, m, vs)

        traj.f_add_parameter_group('individual', 'Contains parameters of the optimizee')

    def create_individual(self):
        """
        Create one individual i.e. one instance of parameters. This instance must be a dictionary with dot-separated
        parameter names as keys and parameter values as values. This is used by the optimizers via the
        function create_individual() to initialize the individual/parameters. After that, the change in parameters is
        model specific e.g. In simulated annealing, it is perturbed on specific criteria
        :return dict: A dictionary containing the names of the parameters and their values
        """

    def simulate(self, traj):
        """
        This is the primary function that does the simulation for the given parameter given (within :obj:`traj`)
        :param  ~l2l.utils.trajectory.Trajectory traj: The trajectory that contains the parameters and the
            individual that we want to simulate. The individual is accessible using `traj.individual` and parameter e.g.
            param1 is accessible using `traj.param1`
        :return: a :class:`tuple` containing the fitness values of the current run. The :class:`tuple` allows a
            multi-dimensional fitness function.
        """
        cell = arb.cable_cell(self.morphology, labels)
        cell.compartments_length(20)

        cell.set_properties(tempK=self.defaults.tempK, Vm=self.defaults.Vm, cm=self.defaults.cm, rL=self.defaults.rL)
        for region, vs in regions:
            cell.paint(f'"{region}"', tempK=vs.tempK, Vm=vs.Vm, cm=vs.cm, rL=vs.rL)
        for region, ion, e in ions:
            cell.paint(f'"{region}"', ion, rev_pot=e)
        cell.set_ion('ca', int_con=5e-5, ext_con=2.0, method=arb.mechanism('nernst/x=ca'))

        for region, mech, values in mechanisms:
            cell.paint(f'"{region}"', arb.mechanism(mech, values))

        cell.place('"center"', arb.iclamp(200, 1000, 0.15))
        model = arb.single_cell_model(cell)
        model.probe('voltage', '"center"', frequency=200000)
        model.properties.catalogue = arb.allen_catalogue()
        model.properties.catalogue.extend(arb.default_catalogue(), '')
        model.run(tfinal=1400, dt=0.005)
        voltages  = np.array(model.traces[0].value[:])
        return (((voltages - reference)**2).sum(), )

# Da Plan
# - make a cell from
#   - random initial params (that's forgetting all the Allen stuff, just keep the mechs/ions, but not parameters)
#   - the morphology (.swc)
# - run the simulation
# - compare U-trace to `nrn.csv`
