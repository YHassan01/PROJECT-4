
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

class SG_14222(GenericWindTurbine):
    def __init__(self):
        """
        paramiters
        __________
        The turbulance intesity varies around 6-8%
        """
        # GenericWindTurbine.__init__(self, name = 'SG 14.0-222DD', diameter = 222,hub_height = 150,
        #                                Power_norm = 14000, turbulance_intesity = 0.07)
        GenericWindTurbine.__init__(self, name='SG 14.0-222DD', diameter=222, hub_height=150, 
                                    power_norm=14000, turbulence_intensity=0.07)


class coastalvirginiawind(UniformWeibullSite):
    def __init__(self, ti= 0.07, shear=PowerShear(h_ref=150, alpha = 0.1)):
        f = [9.1938, 9.9099, 9.0817, 5.2505, 4.8252, 5.7245, 
             11.491, 14.2491, 9.3086, 5.06, 6.4652, 9.4405]
        a = [10.50, 9.94, 8.96, 8.22, 7.34, 7.94,
             11.27, 13.33, 11.86, 10.03, 10.26,  11.12]
        k = [2.260, 2.139, 1.971, 1.771, 1.521,  1.514,
             1.955, 2.568, 2.775, 2.049, 1.951,  2.295]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([xinit, yinit]).T
        self.name = 'Reovolution South Fork Wind'


wind_turbines = SG_14222()

site = coastalvirginiawind()

sim_res = Bastankhah_PorteAgel_2014(site, wind_turbines, k=0.0324555)

def aep_func(x,y):
    aep = sim_res(x,y).aep().sum()
    return aep

def daep_func(x,y):
    daep = sim_res.aep_gradients(gradient_method=autograd, wrt_arg=['x','y'], x=x,y=y)
    return daep


boundary_closed = np.vstack([boundary, boundary[0]])

cost_comp = CostModelComponent(input_keys=['x', 'y'],
                                          n_wt = len(xinit),
                                          cost_function = aep_func,
                                          cost_gradient_function=daep_func,
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
print("Total AEP production of Coastal Virginia: %f MWh"%aep)


# Deliveriable 2
cost, state, recorder = problem.optimize()

recorder.save('optimization_coastalvirginiawind')