import numpy as np

def hessian(chemical_potentials, composition_set):
    '''
    Returns the hessian of the objective function for a single phase

    For the Lagrangian function
        L = N * G + sum(mu_A * (N_A - N * dM_A/dy_i)) + sum(lambda_s * (1 - sum(y_i)))
    We have 5 derivatives
    d2L/dyi2 = N * d2G/dyi2
    d2L/dyidlambda_s = -1 if y_i in s else 0
    d2L/dyidN = dG/dy - sum(mu_A * dM_A/dy_i)
    d2L/dyidmu_A = -N dM_A/dy_i
    d2L/dmu_AdN = -M_A

    Everything is per mole of formula unit, so N has to be corrected for phases where the 
    total moles of atoms could be off from 1

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
    mu = np.asarray(chemical_potentials)

    #dM_A / dy_i
    dxdy = np.zeros((len(elements), len(composition_set.dof)))
    #M_A
    moleA = np.zeros((len(elements),1))
    for comp_idx in range(len(elements)):
        composition_set.phase_record.formulamole_grad(dxdy[comp_idx, :], composition_set.dof, comp_idx)
        composition_set.phase_record.formulamole_obj(moleA[comp_idx,:], composition_set.dof, comp_idx)

    #Moles of phase per formula unit
    #We assume 1 mole of phase, but this is per mole of atoms
    #This is generally okay, but for interstitials or vacancies in the main sublattice
    #We need to use moles of formula units when constructing the hessian
    formulaPhAmt = 1 / np.sum(moleA)

    #dG/dy_i and d2G/dy2
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
    # wrt phase dof - d2L / dyi dyj
    hess[:phase_dof, :phase_dof] = d2g[composition_set.phase_record.num_statevars:,
                                       composition_set.phase_record.num_statevars:] * formulaPhAmt
    
    # wrt phase amount - d2L / dyi dN
    for i in range(phase_dof):
        hess[i, phase_dof] = dg[num_statevars + i] - np.sum(mu * dxdy[:, num_statevars+i])
        hess[phase_dof, i] = hess[i, phase_dof]

    # d2L / dyi dlambda
    cons_jac_tmp = np.zeros((num_internal_cons, len(composition_set.dof)))
    composition_set.phase_record.internal_cons_jac(cons_jac_tmp, composition_set.dof)
    hess[:phase_dof, phase_dof+1:phase_dof+1+num_internal_cons] = -cons_jac_tmp[:, num_statevars:].T
    hess[phase_dof+1:phase_dof+1+num_internal_cons, :phase_dof] = hess[:phase_dof, phase_dof+1:phase_dof+1+num_internal_cons].T

    # d2L / dyi dmuA
    index = phase_dof + num_internal_cons + 1
    hess[:phase_dof, index:] = -1 * dxdy[:, num_statevars:].T * formulaPhAmt
    hess[index:, :phase_dof] = -1 * dxdy[:, num_statevars:] * formulaPhAmt

    # d2L / dmuA dN
    for A in range(len(elements)):
        hess[phase_dof, index + A] = -moleA[A,0]
        hess[index + A, phase_dof] = -moleA[A,0]
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

    Rows correspond to mu_A and columns correspond to X_A so for ternary system with (A,B,R), its
    |   dmu_A/dX_A   dmu_A/dX_B   |
    |   dmu_B/dX_A   dmu_B/dX_B   |

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

    Rows correspond to mu_A and columns correspond to X_A so for binary system, its
    |   dmu_A/dX_A   dmu_A/dX_B   |
    |   dmu_B/dX_A   dmu_B/dX_B   |

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
