from pysb.simulator import ScipyOdeSimulator
import numpy as np
from earm2_flat import model
import matplotlib.pyplot as plt
from equilibration_function import pre_equilibration

fitted_pars = np.load('calibrated_6572pars.npy')
par = fitted_pars[0]

par_eq = np.copy(par)
par_eq[0] = 0
time_eq = np.linspace(0, 10000, 100)
initials = pre_equilibration(model, time_eq, par_eq)[1]
initials[0][0] = 100

tspan = np.linspace(0, 20000, 100)
sim1 = ScipyOdeSimulator(model, tspan).run(param_values=par, initials=initials[0]).all

par[83] = par[83] * .2
par[85] = par[85] * .2
par[87] = par[87] * .2
# par[82] = par[82] * 1
# par[84] = par[84] * 1
# par[86] = par[86] * 1
# initials[0][6] = 0
sim2 = ScipyOdeSimulator(model, tspan).run(param_values=par, initials=initials[0]).all


plt.plot(tspan, sim1['__s64'], label='WT')
plt.plot(tspan, sim2['__s64'], label='Mutation')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Population')
plt.show()