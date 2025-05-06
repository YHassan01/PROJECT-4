import numpy as np

from topfarm.cost_models.cost_model_wrappers import CostModelComponent
from topfarm import TopFarmProblem
from topfarm.plotting import NoPlot, XYPlotComp
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm.constraint_components.spacing import SpacingConstraint
import topfarm

from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014, Zong_PorteAgel_2020, Niayifar_PorteAgel_2016, CarbajoFuertes_etal_2018, Blondel_Cathelain_2020
from py_wake.utils.gradients import autograd
from py_wake.site._site import UniformWeibullSite
from py_wake.wind_turbines.generic_wind_turbines import GenericWindTurbine
from py_wake.site.shear import PowerShear
import pickle


with open('utm_boundary.pkl', 'rb') as f:
    boundary = np.array(pickle.load(f))

with open('utm_layout.pkl', 'rb') as f:
    xinit,yinit = np.array(pickle.load(f))


maxiter = 1000
tol = 1e-6

class SG_110_200_DD(GenericWindTurbine):
    def __init__(self):
        """
        Parameters
        ----------
        The turbulence intensity Varies around 6-8%
        Hub Height Site Specific
        """
        GenericWindTurbine.__init__(self, name='SG 11.0-200 DD', diameter=200, hub_height=150,
                             power_norm=11000, turbulence_intensity=0.08)


class Revolutionwind_southforkwind(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [5.1037, 7.9759, 5.6619, 5.8254, 5.4211, 6.2595, 10.1765, 9.5321, 16.8622, 11.1874, 8.6163, 7.3778] 
        a = [9.46, 10.86,9.04, 9.17, 9.51, 10.00, 9.87, 12.15, 2.64,11.75, 9.70, 9.55] 
        k = [2.111, 2.404, 2.381, 2.463, 2.404, 2.029, 2.084, 3.225, 3.295, 2.912, 2.244, 1.967] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([xinit, yinit]).T
        self.name = "Karehamn"


wind_turbines = SG_110_200_DD()

site = Revolutionwind_southforkwind()

sim_res = Bastankhah_PorteAgel_2014(site, wind_turbines, k=0.0324555)

def aep_func(x,y):
    aep = sim_res(x,y).aep().sum()
    return aep


boundary_closed = np.vstack([boundary, boundary[0]])


cost_comp = CostModelComponent(input_keys=['x', 'y'],
                                          n_wt = len(xinit),
                                          cost_function = aep_func,
                                          objective=True,
                                          maximize=True,
                                          output_keys=[('AEP', 0)]
                                          )


problem = TopFarmProblem(design_vars= {'x': xinit, 'y': yinit},
                         constraints=[XYBoundaryConstraint(boundary),
                                      SpacingConstraint(334)],
                        cost_comp=cost_comp,
                        driver=EasyScipyOptimizeDriver(optimizer='SLSQP', maxiter=maxiter, tol=tol),
                        n_wt=len(xinit),
                        expected_cost=0.001,
                        plot_comp=XYPlotComp()
                        )


# Deliveriable 1
aep = sim_res(xinit,yinit).aep().sum()
print("Total AEP production of Karehamn: %f MWh"%aep)


# Deliveriable 2
cost, state, recorder = problem.optimize()

recorder.save('optimization_Karehamn')