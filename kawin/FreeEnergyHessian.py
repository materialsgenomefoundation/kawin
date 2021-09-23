import numpy as np
from pycalphad import Model, variables as v
from pycalphad.codegen.callables import build_callables

setattr(v, 'GE', v.StateVariable('GE'))

def hessian(chemical_potentials, composition_set):
    '''
    Returns the hessian of the objective function for a single phase

    Parameters
    ----------
    chemical_potentials : 1-D ndarray
    composition_set : pycalphad.core.composition_set.CompositionSet
    
    Returns
    -------
    Matrix of floats for each second derivative
    Derivatives along each axis will be in order of:
        site fractions, phase amount, lagrangian multipliers, chemical potential
    '''
    elements = list(composition_set.phase_record.nonvacant_elements)
    x = np.array(composition_set.X)
    mu = np.asarray(chemical_potentials)
    dxdy = np.zeros((len(elements), len(composition_set.dof)))
    for comp_idx in range(len(elements)):
        composition_set.phase_record.formulamole_grad(dxdy[comp_idx, :], composition_set.dof, comp_idx)
    dg = np.zeros(len(composition_set.dof))
    composition_set.phase_record.formulagrad(dg, composition_set.dof)
    d2g = np.zeros((len(composition_set.dof), len(composition_set.dof)))
    composition_set.phase_record.formulahess(d2g, composition_set.dof)

    phase_dof = composition_set.phase_record.phase_dof
    num_internal_cons = composition_set.phase_record.num_internal_cons
    num_statevars = composition_set.phase_record.num_statevars
    #Create hessian matrix
    hess = np.zeros((phase_dof + num_internal_cons + len(elements) + 1,
                     phase_dof + num_internal_cons + len(elements) + 1))
    # wrt phase dof
    hess[:phase_dof, :phase_dof] = d2g[composition_set.phase_record.num_statevars:,
                                       composition_set.phase_record.num_statevars:]
    cons_jac_tmp = np.zeros((num_internal_cons, len(composition_set.dof)))
    composition_set.phase_record.internal_cons_jac(cons_jac_tmp, composition_set.dof)

    # wrt phase amount
    for i in range(phase_dof):
        hess[i, phase_dof] = dg[num_statevars + i] - np.sum(mu * dxdy[:, num_statevars+i])
        hess[phase_dof, i] = hess[i, phase_dof]

    hess[:phase_dof, phase_dof+1:phase_dof+1+num_internal_cons] = -cons_jac_tmp[:, num_statevars:].T
    hess[phase_dof+1:phase_dof+1+num_internal_cons, :phase_dof] = hess[:phase_dof, phase_dof+1:phase_dof+1+num_internal_cons].T
    index = phase_dof + num_internal_cons + 1
    hess[:phase_dof, index:] = -1 * dxdy[:, num_statevars:].T
    hess[index:, :phase_dof] = -1 * dxdy[:, num_statevars:]

    for A in range(len(elements)):
        hess[phase_dof, index + A] = -x[A]
        hess[index + A, phase_dof] = -x[A]
    return hess


def totalddx(chemical_potentials, composition_set, refElement):
    '''
    Total derivative of site fractions, phase amount, lagrangian multipliers 
    and chemical potential with respect to system composition
    d/dx = partial d/dxA - partial d/dxR where R is reference

    Parameters
    ----------
    chemical_potentials : 1-D ndarray
    composition_set : pycalphad.core.composition_set.CompositionSet
    refElement : str
        Reference element
    
    Returns
    -------
    Array of floats for each derivative
    Derivatives will be in order of:
        site fractions, phase amount, lagrangian multipliers, chemical potential
    '''
    elements = list(composition_set.phase_record.nonvacant_elements)
    b = np.zeros((composition_set.phase_record.phase_dof + composition_set.phase_record.num_internal_cons + len(elements) + 1, len(elements) - 1))
    i0 = composition_set.phase_record.phase_dof + composition_set.phase_record.num_internal_cons + 1

    c = 0
    for A in range(len(elements)):
        if elements[A] != refElement:
            b[i0 + A, c] = -1
            c += 1
        else:
            b[i0 + A, :] = 1

    #If curvature is undefined, then assume inverse is 0
    #For derivatives with respect to composition, this means that the internal DOFs of the phase is independent of composition
    try:
        inverse = np.linalg.inv(hessian(chemical_potentials, composition_set))
        return np.matmul(inverse, b)
    except:
        return np.zeros(b.shape)


def partialddx(chemical_potentials, composition_set):
    '''
    Partial derivative of site fractions, phase amount, lagrangian multipliers 
    and chemical potential with respect to system composition

    Parameters
    ----------
    chemical_potentials : 1-D ndarray
    composition_set : pycalphad.core.composition_set.CompositionSet
    
    Returns
    -------
    Array of floats for each derivative
    Derivatives will be in order of:
        site fractions, phase amount, lagrangian multipliers, chemical potential
    '''
    elements = list(composition_set.phase_record.nonvacant_elements)
    b = np.zeros((composition_set.phase_record.phase_dof + composition_set.phase_record.num_internal_cons + len(elements) + 1, len(elements)))
    i0 = composition_set.phase_record.phase_dof + composition_set.phase_record.num_internal_cons + 1

    for A in range(len(elements)):
        b[i0 + A, A] = -1

    #If curvature is undefined, then assume inverse is 0
    #For derivatives with respect to composition, this means that the internal DOFs of the phase is independent of composition
    try:
        inverse = np.linalg.inv(hessian(chemical_potentials, composition_set))
        return np.matmul(inverse, b)
    except np.linalg.LinAlgError:
        return np.zeros(b.shape)


def dMudX(chemical_potentials, composition_set, refElement):
    '''
    Total derivative of chemical potential with respect to system composition
    dmuA/dxB = (partial dmuA/dxB - partial dmuA/dxR) - (partial dmuR/dxB - partial dmuR/dxR) 
    where R is reference

    This more or less represents the curvature of the free energy surface with reference element R

    Parameters
    ----------
    chemical_potentials : 1-D ndarray
    composition_set : pycalphad.core.composition_set.CompositionSet
    refElement : str
        Reference element
    
    Returns
    -------
    Array of floats for each derivative, (n-1 x n-1) matrix
    Derivatives will be in alphabetical order of elements
    '''
    ddx = totalddx(chemical_potentials, composition_set, refElement)
    i0 = composition_set.phase_record.phase_dof + composition_set.phase_record.num_internal_cons + 1
    elements = list(composition_set.phase_record.nonvacant_elements)
    
    dmudx = np.zeros((len(elements) - 1, len(elements) - 1))

    c = 0
    for A in range(len(elements)):
        if elements[A] != refElement:
            dmudx[c, :] += ddx[i0 + A, :]
            c += 1
        else:
            for B in range(len(elements) - 1):
                dmudx[B, :] -= ddx[i0 + A, :]

    return dmudx

def partialdMudX(chemical_potentials, composition_set):
    '''
    Partial derivative of chemical potential with respect to system composition

    Parameters
    ----------
    composition_set : pycalphad.core.composition_set.CompositionSet
    
    Returns
    -------
    Array of floats for each derivative, (n x n) matrix
    Derivatives will be in alphabetical order of elements
    '''
    ddx = partialddx(chemical_potentials, composition_set)
    i0 = composition_set.phase_record.phase_dof + composition_set.phase_record.num_internal_cons + 1
    
    return ddx[i0:,:]
