import sympy as sp
from sympy import Symbol, Eq, Abs, Function

import modulus
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_1d import Point1D, Line1D
from modulus.sym.eq.pde import PDE
from modulus.sym.key import Key
from modulus.sym.models.fully_connected import FullyConnectedArch
from modulus.sym.domain.constraint import PointwiseBoundaryConstraint, PointwiseInteriorConstraint
from modulus.sym.node import Node
from modulus.sym.geometry.parameterization import Parameterization
from modulus.sym.hydra.config import ModulusConfig 
from modulus.sym.hydra.utils import to_absolute_path, instantiate_arch

from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.domain.inferencer import PointwiseInferencer
from modulus.sym.utils.io import (
    csv_to_dict, ValidatorPlotter, InferencerPlotter
)
from modulus.sym.models.activation import Activation
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import os
import torch
from torchviz import make_dot

# Constants
V = 100  # L
q = 100 # L/min
C_A0 = 1.0  # mol/L
T_0 = 350  # K
V = 100    #L
T_c = 300  # K
rho = 1000  # g/L
C_p = 0.239  # J/kg.K
k_0 = 7.2E10  # min^-1
#E = 50000  # J/mol
#R = 8.314  # J/mol.K
EoverR = 8750  #k
delta_H_r = -50000  # J/mol
#U = 200  # J/min.K
#A = 10  # m²
UA = 5E4   #J/min K
t, T_c = Symbol("t"), Symbol("T_c")
C_A, T, k = Symbol("C_A"), Symbol("T"), Symbol("k")
#parameter_range = {t: (0, 10), T_c: (290,310)}
input_variables = {"t" : t, "T_c": T_c}

# define custom class
class ReactorPlotter(InferencerPlotter):
    def __call__(self, invar, outvar):
        # get input variables
        t, T_c = invar["t"][:,0], invar["T_c"][:,0]
        #print("shape", t.shape, T_c.shape)
        C_A, T, k = outvar["C_A"], outvar["T"], outvar["k"]
        #print(C_A, T)
        
        # make plot
        plt.figure(figsize=(20,5), dpi=100)
        plt.suptitle("CSTR Reactor")
        fig, ax=plt.subplots(2,1)
        ax[0].plot(t, C_A)
        ax[0].set_xlim([0, 10])
        #ax[0].set_ylim([0, 1])
        ax[0].set_title("Reactant Concentration (mol/L)")
        
        ax[1].plot(t, T)
        ax[1].set_title("Reactor Temp (K)")
        ax[1].set_xlim([0, 10])
        #ax[1].set_ylim([300, 450])
        plt.tight_layout()

        return [(fig, "custom_plot"),]


class CSTR_Reactor(PDE):
    def def_geometry(self):
        self.geo=Line1D(0.,10.)
        #self.geo=Point1D(0.0)
        return

    def def_nodes(self):
        u_net = instantiate_arch(
            input_keys=[Key("t"), Key("T_c")],
            #input_keys=[Key("t")],
            output_keys=[Key('C_A'), Key('T'), 
                         Key('k'), Key('material_balance'), Key('energy_balance')],
            #cfg=self.cfg.arch.modified_fourier,
            cfg=self.cfg.arch.fully_connected,
            #activation_fn=Activation.RELU,
            #nr_layers=10,
        )

        #model_2 = instantiate_arch(
        #    input_keys=[Key("x"), Key("y")],
        #    output_keys=[Key("u"), Key("v")],
        #    cfg=cfg.arch.siren,
        #)

        #u_net = FullyConnectedArch(input_keys=[Key('t'), Key('T_c')],
        #                           output_keys=[Key('C_A'), Key('T'),
        #                                        Key('k'), Key('material_balance'), Key('energy_balance')],
        #                           nr_layers=10, layer_size=256)
        self.nodes = [u_net.make_node(name="CSTR")]

        #visualize the network
        print(self.nodes)

        # graphically visualize the PyTorch execution graph
        # NOTE: Requires installing torchviz library: https://pypi.org/project/torchviz/

        # pass dummy data through the model
        data_out = u_net({"t": (torch.rand(1)), "T_c": (torch.rand(1))})
        make_dot(data_out["C_A"], params=dict(u_net.named_parameters())).render("reactor-C_A", format="png")
        make_dot(data_out["T"], params=dict(u_net.named_parameters())).render("reactor-T", format="png")
        return

    def def_constraints(self):
        tparam = np.linspace(0,10,1000).reshape(-1,1)    # For 10 min and split it into 1000 pieces 
        tcparam = np.linspace(290,310,1).reshape(-1,1)   # For 290-310K every Kelvin
        #print(tparam.shape, tcparam.shape)
        prange = {'t': tparam, 'T_c': tcparam}

        # Initial conditions
        IC = PointwiseBoundaryConstraint(
            nodes=self.nodes,
            outvar={'C_A': C_A0, 'T': 350},
            batch_size=2,
            #criteria=sp.Eq(time_domain.get_variable(), 0),
            #lambda_weighting={'C_A': 1.0, 'T': 1.0},
            geometry=self.geo,
            parameterization={'t': 0.0, 'T_c':300},
            shuffle=False,
        )
        self.domain.add_constraint(IC, "initial_conditions")

        #Interior constraints (PDE)
        pdeconstraint = PointwiseInteriorConstraint(
            nodes=self.nodes,
            outvar={"material_balance":0.0, "energy_balance":0.0, "k": 0.0},
            batch_size=1000,
            #bounds=m,
            #criteria=,
            geometry=self.geo,
            parameterization={'t': (0.0,10.0), 'T_c':(290,310)},
            shuffle= False,
        )
        self.domain.add_constraint(pdeconstraint, "pde")

        return

    def def_domain(self):
        self.domain = Domain()
        return

    def def_solver(self):
        print("In solver")
        # Solver
        self.slv = Solver(self.cfg,
            domain=self.domain,
            #nodes=self.nodes,
            #lr=1e-3,
            #epochs=1000,
            #monitor=None
        )

        # Train
        self.slv.solve()

        return

    def def_fun_and_eq(self):
        self.C_A = Function("C_A")(*input_variables)
        self.T = Function("T")(*input_variables)
        self.k = Function("k")("T")

        self.equations = {}
        # Reaction rate
        self.equations["k"]= (k - (k_0 * sp.exp(-(EoverR/T))))**2

        # Material balance for A
        self.equations["material_balance"] = (C_A.diff(t) - ((q / V) * (C_A0 - C_A) - k * C_A))**2
        
        #dC_A_dt = (F / V) * (C_A0 - C_A) - k * C_A
        #print(self.equations["material_balance"])
        # Energy balance
        self.equations["energy_balance"] = (T.diff(t) - ((q / V) * (T_0 - T) +
                                                        ((-delta_H_r * k * C_A)/ (rho * C_p)) +
                                                        ((UA *(T_c-T)) / (rho * C_p * V))))**2
        print(self.equations)
        return

    def def_inference(self):
        # add inferencer data
        tdict=np.linspace(0, 10, 1000).reshape(-1,1)
        tcdict=np.full((1000), 290.).reshape(-1,1)
        self.inf290 = PointwiseInferencer(
            nodes=self.nodes,
            invar={'t': tdict, 'T_c': tcdict},
            output_names=["C_A", "T", "k"],
            requires_grad=False,
            plotter=ReactorPlotter(),
        )
        self.domain.add_inferencer(self.inf290, "Reactor-290")

        tcdict=np.full((1000), 300.).reshape(-1,1)
        self.inf300 = PointwiseInferencer(
            nodes=self.nodes,
            invar={'t': tdict, 'T_c': tcdict},
            output_names=["C_A", "T", "k"],
            requires_grad=False,
            plotter=ReactorPlotter(),
        )
        self.domain.add_inferencer(self.inf300, "Reactor-300")
        
        tcdict=np.full((1000), 305.).reshape(-1,1)
        self.inf305 = PointwiseInferencer(
            nodes=self.nodes,
            invar={'t': tdict, 'T_c': tcdict},
            output_names=["C_A", "T", "k"],
            requires_grad=False,
            plotter=ReactorPlotter(),
        )
        self.domain.add_inferencer(self.inf305, "Reactor-305")

    #def def_mon(self):
    #    return
    
    #def def_valid(self):
        # add validator
    #    file_path = "cstr-valid.csv"
    #    if os.path.exists(to_absolute_path(file_path)):
    #        mapping = {"Tc(290)": "T_c", "Ca(290)": "C_A", "Tc(300)": "T_c300", "Ca(300)": "C_A300", "Tc(305)": "T_c305", "Ca(305)": "C_A305"}
    #        cstr_data = csv_to_dict(to_absolute_path(file_path), mapping)
    #        cstr_invar_numpy = {
    #            key: value for key, value in cstr_data.items() if key in ["T_c", "C_A"]
    #        }
    #        
    #        cstr_outvar_numpy = {
    #            key: value for key, value in cstr_data.items() if key in ["T_c", "C_A"]
    #        }
            
    #        self.validator = PointwiseValidator(
    #            nodes=self.nodes,
    #            invar=cstr_invar_numpy,
    #            true_outvar=cstr_outvar_numpy,
    #            batch_size=1024,
    #            plotter=ValidatorPlotter(),
    #        )
    #        self.domain.add_validator(self.validator)
            
    def __init__(self, cfg):
        self.cfg = cfg
        self.def_fun_and_eq()
        self.def_domain()
        self.def_nodes()
        self.def_geometry()
        self.def_constraints()
        self.def_inference()
        self.def_solver()

        #evaluate
        #print("Evaluating...")
        #self.slv.eval()

        return

@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    # Make list of nodes to unroll graph on
    reactor = CSTR_Reactor(cfg)
    #reactor.infer()
    return

if __name__ == "__main__":
   run()
