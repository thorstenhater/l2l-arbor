import utils, arbor as arb

# !\circled{1}! read in geometry
segment_tree = arb.load_swc_allen('cell.swc', no_gaps=False)
morphology = arb.morphology(segment_tree)
# !\circled{2}! assign names to regions defined by SWC and center of soma
labels = arb.label_dict({'soma': '(tag 1)', 'axon': '(tag 2)',
                         'dend': '(tag 3)', 'apic': '(tag 4)',
                         'center': '(location 0 0.5)'})
cell = arb.cable_cell(morphology, labels) # see !\circled{3}!
cell.compartments_length(20) # discretisation strategy: max compartment length
# !\circled{4}! load and assign electro-physical parameters
defaults, regions, ions, mechanisms = utils.load_allen_fit('fit.json')
# set defaults and override by region
cell.set_properties(tempK=defaults.tempK, Vm=defaults.Vm,
                    cm=defaults.cm, rL=defaults.rL)
for region, vs in regions:
    cell.paint('"'+region+'"', tempK=vs.tempK, Vm=vs.Vm, cm=vs.cm, rL=vs.rL)
# set reversal potentials
for region, ion, e in ions:
    cell.paint('"'+region+'"', ion, rev_pot=e)
cell.set_ion('ca', int_con=5e-5, ext_con=2.0, method=arb.mechanism('nernst/x=ca'))
# assign ion dynamics
for region, mech, values in mechanisms:
    cell.paint('"'+region+'"', arb.mechanism(mech, values))
# !\circled{5}! attach stimulus and spike detector
cell.place('"center"', arb.iclamp(200, 1000, 0.15))
cell.place('"center"', arb.spike_detector(-40))
# !\circled{6}! set up runnable simulation
model = arb.single_cell_model(cell)
model.probe('voltage', '"center"', frequency=200000) # see !\circled{5}!
# !\circled{7}! assign catalogues
model.properties.catalogue = arb.allen_catalogue()
model.properties.catalogue.extend(arb.default_catalogue(), '')
# !\circled{8}! run simulation and plot results
model.run(tfinal=1400, dt=0.005)
utils.plot_results(model)
