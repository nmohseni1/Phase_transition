import dimod
import qiskit
from qiskit.utils import QuantumInstance
from qiskit import *

from qiskit.algorithms import QAOA
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from qiskit import Aer
from qiskit.opflow import I, X, Y, Z, Zero, One, PauliExpectation, CircuitSampler, StateFn, DictStateFn, CircuitStateFn
from qiskit.algorithms.optimizers import COBYLA, ADAM, SLSQP
from qiskit.tools.visualization import plot_histogram
from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import TransformationPass
from qiskit.test.mock import FakeGuadalupe
from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.algorithms import RecursiveMinimumEigenOptimizer, MinimumEigenOptimizer
from qiskit.providers.aer import AerSimulator
from mitiq.zne import execute_with_zne
from mitiq.zne import scaling
from mitiq.zne.inference import AdaExpFactory, ExpFactory, RichardsonFactory, LinearFactory, PolyFactory
import qiskit.providers.aer.noise as noise
from qiskit.algorithms.optimizers import SPSA
from qiskit.algorithms import NumPyMinimumEigensolver,NumPyEigensolver
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit import QuantumCircuit, execute, Aer
from scipy.optimize import minimize
from typing import Iterable
import scipy
import operator
from scipy.optimize.zeros import RootResults
from SAT import *
ideal_backend = Aer.get_backend('qasm_simulator')
class Hamiltonian:
    """
   Ksat=True means1-2-SAT
   Ksat=False means1-3-SAT
    """
    
    
    def __init__(self,n,m,ksat=False):
        """
        d < n 
        d*n even
        """
        self.n = n
        self.nqubits=n
        self.m = m
        self.ksat = ksat
        #self.J = {pair: 2*np.random.randint(2)-1 for pair in list(self.G.edges())}
        #self.J = {pair: 2*np.random.rand()-1 for pair in list(self.G.edges())}
        self.pols = None
        if ksat:
            #print('Local terms are not supported by BP!')
            self.J,self.A=get_coupling_J_one_two_sat(n,m)
            self.h = np.zeros(n)
        else:
            self.h,self.J,self.A= get_coupling_J_one_three_sat(n,m)
        

    def drawGraph(self):
        pos = nx.spring_layout(self.G)
        nx.set_edge_attributes(self.G, values = self.J, name = 'weight')
        labels = nx.get_edge_attributes(self.G,'weight')
        nx.draw_networkx_edge_labels(self.G,pos ,edge_labels=labels)
        return nx.draw(self.G,pos=pos, with_labels=True, alpha=0.8, node_size=500)

    def setCoupling(self,J):
        for pair in list(self.G.edges()):
            self.J.update({pair:J})

    def pol(self):
        """returns list of al polarization operators."""
        return [(I^j)^Z^(I^(self.nqubits-j-1)) for j in range(self.n)]
    
    def cor(self):
        """returns list of all correlation operators in Hamiltonian"""
        return [((I^pair[0])^Z^(I^(self.n-pair[0]-1)))@((I^pair[1])^Z^(I^(self.n-pair[1]-1))) for pair in list(self.J.keys())]

    def operator(self):
        """creates Hamiltonian operator """
        op = I^self.n
        for pair in list(self.J.keys()):
            op += self.J.get(pair)*((I^pair[0])^Z^(I^(self.n-pair[0]-1)))@((I^pair[1])^Z^(I^(self.n-pair[1]-1)))
        for i in range(self.n):
            op += self.h[i]*((I^i)^Z^(I^(self.n-i-1)))
        op -= I^self.n
        return op.reduce()

    def gsDensOp(self):
        "returns the density operator of the exact ground state"
        k = 2
        npme = NumPyEigensolver(k)
        mini=npme.compute_eigenvalues(operator=self.operator())
        while mini.eigenvalues[0] == mini.eigenvalues[-1]:
            k+= 1
            npme = NumPyEigensolver(k)
            mini=npme.compute_eigenvalues(operator=self.operator())
        gsFn=np.sum(mini.eigenstates[0:-1])
        return qiskit.opflow.primitive_ops.MatrixOp(gsFn.to_density_matrix())/len(mini.eigenstates[0:-1])

    def callback(self,neval,param,res,err):
        """Record optimization history, in a global variable rec, which must be initialized"""
        global rec
        rec.append(res)
        return

    def runIdeal(self, p, record=False):
        """returns optimization result of QAOA and the QAOA instance"""
        callback = None
        if record:
            callback = self.callback
        solver = QAOA(COBYLA(maxiter=50000),reps=p,quantum_instance=QuantumInstance(Aer.get_backend('statevector_simulator')),initial_point=np.append(np.linspace(2,0.1,p),np.linspace(0.5,2.5,p)),callback=callback)
        res = solver.compute_minimum_eigenvalue(self.operator())
        if res.cost_function_evals >= 50000:
            print('not converged',res)
        return res, solver

    def runiiIdeal(self,ps):
        # ps: list of increasing p values
        # returns list of optiization results
        initial = 0.1*np.random.rand(2*ps[0])
        ress = []
        solver = None
        for i in range(len(ps)):
            p = ps[i]
            solver = QAOA(COBYLA(maxiter=50000,rhobeg=1),reps=p,quantum_instance=QuantumInstance(Aer.get_backend('statevector_simulator')),initial_point=initial)
            ress.append(solver.compute_minimum_eigenvalue(self.operator()))
            if i < len(ps) -1 :
                initial=np.concatenate((ress[-1].optimal_point[:p], 
                                        0.1*np.random.rand(ps[i+1]-p),
                                        ress[-1].optimal_point[p:],
                                        0.1*np.random.rand(ps[i+1]-p)
                                        ))
        return ress[-1], solver

    def idealOverl(self,state,res,solver):
        """calculates the overlap of a statefunction with the QAOA ansatz. res contains the values for beta and gamma. Often given as res.x or res.optimal_point when res is porvided by a run method"""
        cir=solver.construct_circuit(res,self.operator())[0]
        return np.abs(((~state @ CircuitStateFn(cir)).eval()))**2

    def idealExp(self,op,res,solver):
        """calculates the ideal expectation value of op shot based."""
        cir=solver.construct_circuit(res,self.operator())[0]
        measurable_expression = StateFn(op, is_measurement=True).compose(CircuitStateFn(cir)) 
        expectation = PauliExpectation().convert(measurable_expression)  
        sampler = CircuitSampler(QuantumInstance(Aer.get_backend('qasm_simulator'),shots=2**15)).convert(expectation) 
        return sampler.eval().real
    
    def stateED(self):
        npme = NumPyEigensolver(2)
        k=2
        mini=npme.compute_eigenvalues(operator=self.operator())
        while mini.eigenvalues[0] == mini.eigenvalues[-1]:
            k += 1
            npme = NumPyEigensolver(k)
            mini=npme.compute_eigenvalues(operator=self.operator())
        return mini
    
    def getPolsIdeal(self,res,solver):
        """get ideal expectation values for all polarizations"""
        ops = [(I^j)^Z^(I^(self.nqubits-j-1)) for j in range(self.nqubits)]
        return [self.idealExp(ops[j],res,solver) for j in range(self.nqubits)]
    
    def getCorsIdeal(self,res,solver):
        """ get ideal expectation values for all correlations"""
        ops = [(I^j)^Z^(I^k)^Z^(I^(self.nqubits-j-k-2)) for j in range(self.nqubits-1) for k in range(self.nqubits-j-1)]
        return [self.idealExp(op,res,solver) for op in ops]

    def runNoisy(self,p,shots=2**15,tol=None,record=False,initial_point=None):
        """run qaoa with noise"""
        callback = None
        if record:
            callback = self.callback
        op = self.operator()
        #print('Hamiltonian:\n',op)
        solver = QAOA(reps=p,quantum_instance=QuantumInstance(ideal_backend,shots=shots,noise_model=noise_model),optimizer=COBYLA(tol=tol,maxiter=50000),initial_point=np.append(np.linspace(2,0.1,p),np.linspace(0.5,2.5,p)),callback=callback)
        res =  solver.compute_minimum_eigenvalue(operator=op)             
        if res.cost_function_evals >= 50000:
            print('not converged',res)
        return res, solver

    def noisyExp(self,op,res,solver,shots=2**15):
        """calculate the noisy expectation value"""
        # define the state to sample
        cir=solver.construct_circuit(res,self.operator())[0]
        backend = Aer.get_backend('aer_simulator')
        cir = transpile(cir, backend)
        cir = RZZTranslator()(cir)
        measurable_expression = StateFn(op, is_measurement=True).compose(CircuitStateFn(cir)) 
        # convert to expectation value
        expectation = PauliExpectation().convert(measurable_expression)  
        # get state sampler (you can also pass the backend directly)
        sampler = CircuitSampler(QuantumInstance(ideal_backend,shots=shots,noise_model=noise_model)).convert(expectation) 
        # evaluate
        return sampler.eval().real
    
def checkIdealQAOA(d,nqubits,p,maxcut):
    mod = dRegularIsing(d,nqubits,localTerm=False)
    if maxcut:
        mod.setCoupling(1)
    npme = NumPyMinimumEigensolver()
    resRef = npme.compute_minimum_eigenvalue(operator=mod.operator())
    gsFn = resRef.eigenstate
    E0 = resRef.eigenvalue
    res, solver = mod.runIdeal(p)
    print(res)
    return mod.idealOverl(gsFn,res.optimal_point,solver), res.eigenvalue.real/E0

def get_coupling_J_one_two_sat(n,m):
    A=P1in2SAT(n,m )
    n, m = A.shape
    pairs = combinations(range(n), 2)
    J={pair: np.sum(A[pair[0]]*A[pair[1]]) for i, pair in enumerate(pairs) if abs(np.sum(A[pair[0]]*A[pair[1]]))>1e-10}
    return(J,A)
def get_coupling_J_one_three_sat(n,m):
    A=P1in3SAT(n,m )
    n, m = A.shape
    h = -0.5*np.sum(A, axis=1)
    #J = np.zeros(n*(n-1)//2)
    pairs = combinations(range(n), 2)

    J={pair: 0.5* np.sum(A[pair[0]]*A[pair[1]]) for i, pair in enumerate(pairs) if abs(np.sum(A[pair[0]]*A[pair[1]]))>1e-10}
    return(h,J,A)
def QAOA_violate(mod,n,m,P,Seed):
    np.random.seed(Seed)
    #mod = Hamiltonian(n,m,ksat=False)
    res, solver = mod.runIdeal(P)
    cir=solver.construct_circuit(res._optimal_parameters,mod.operator())[0]
    state_i=CircuitStateFn(cir).eval()
    num=violateNum_state(state_i, n,mod.A)
    Num_i=((m-num)/m)
        
    return(num)

def violateNum_basis(s, A):
    # claculate the number of violated clauses given a basis bit string s
    n, m = A.shape
    v_choice = [int(x) for x in s] # the assignment of variables
    violate_num = 0
    res = np.dot(v_choice, A)
    violate_num = len(res[res != 1])
    return violate_num


def violateNum_state(State, n,A):
    # prepare the state
    state = State._primitive._data
    # expected number of violated clauses given a state
    pr = np.abs(state)**2
    nums = np.zeros(len(pr), dtype=int) # number of violated clauses for each basis
    for k in range(len(pr)):
        s = np.binary_repr(k, n)[::1]
        nums[k] = violateNum_basis(s, A)
    #print('pr',pr)
    #print('nums',nums)
   # print(sum(pr*nums))
    return sum(pr*nums)
   
