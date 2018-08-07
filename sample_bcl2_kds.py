import numpy as np
from pysb.simulator import CupSodaSimulator
from earm2_flat import model
import numpy as np
import pickle
from tropical.dynamic_signatures_range import run_tropical_multi

kr_pars_to_sample = [71, 74, 83, 85, 87, 89, 91, 93, 95, 97, 99, 101]
kf_pars_to_calculate = [70, 73, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100]


def sample_kr_uniform(size):
    samples = np.random.uniform(low=1E-3, high=1E-2, size=size)
    return samples


def sample_kd_uniform(parameter_idx, size):
    if parameter_idx in [71, 89, 97, 99]:
        samples = np.random.uniform(low=60221.4086, high=301107.043, size=size)
        return samples
    elif parameter_idx in [83, 85, 87, 93, 95, 101]:
        samples = np.random.uniform(low=602.214086, high=60221.4086, size=size)
        return samples
    elif parameter_idx in [74]:
        samples = np.random.uniform(low=1806642.258, high=6022140.86, size=size)
        return samples
    elif parameter_idx in [91]:
        samples = np.random.uniform(low=602214.086, high=1806642.258, size=size)
        return samples
    else:
        raise ValueError('Distribution is not defined for parameter index {}'.format(parameter_idx))


parameters = np.array([p.value for p in model.parameters])

nsamples = 50000
repeated_parameter_values = np.tile(parameters, (nsamples, 1))

for par_idx in kr_pars_to_sample:
    repeated_parameter_values[:, par_idx] = sample_kr_uniform(nsamples)

for kf_idx, kr_idx in zip(kf_pars_to_calculate, kr_pars_to_sample):
    repeated_parameter_values[:, kf_idx] = repeated_parameter_values[:, kr_idx] / sample_kd_uniform(kr_idx, nsamples)
np.save('pars_sampled_kd_bcl2.npy', repeated_parameter_values)
#
tspan = np.linspace(0, 20000, 100)

vol= 1e-19
integrator_opt = {'rtol': 1e-6, 'atol': 1e-6, 'mxsteps': 20000}
sims = CupSodaSimulator(model, tspan=tspan, gpu=0, memory_usage='shared_constant', vol=vol,
                        integrator_options=integrator_opt).run(param_values=repeated_parameter_values)
sims.save('simulations_sampled_kd_bcl2.h5')

signatures = run_tropical_multi(model=model, simulations=sims, cpu_cores=20, verbose=True)

with open('signatures_sampled_kd_bcl2.pickle', 'wb') as handle:
    pickle.dump(signatures, handle, protocol=pickle.HIGHEST_PROTOCOL)