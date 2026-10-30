import numpy as np

def get_jac(dom, cod): # Dummy
    return np.random.randint(100, size=cod * dom).reshape(cod, dom)

def get_hes(dom, cod): # Dummy
    hes = np.random.randint(100, size=cod * (dom ** 2)).reshape(dom, cod, dom)
    return (hes + hes.T) / 2    # hes.T = hes.transpose(2, 1, 0)

def random_dimension():
    return np.random.randint(low=1, high=10)

def first_info(dom, cod, jac, hes):
    print(f'T0: M0 ({dom}) -> M1 ({cod}) jac: ({dom}, {dom}) hes: ({dom}, {dom}, {dom})',
    '->', 'jac:', jac.shape, 'hes:', hes.shape)
def prev_info(i, dom, cod, jac, hes):
    print(f'T{i}: M{i} ({dom}) -> M{i+1} ({cod}) jac:', jac.shape, 'hes:', hes.shape, end=' -> ')
def next_info(jac, hes):
    print('jac:', jac.shape, 'hes:', hes.shape)

dom = random_dimension() # Dimension of domain   of T0, hence M_0
cod = random_dimension() # Dimension of codomain of T0, hence M_1

# Jacobian matrix (dummy) in M_1, M_0*
jac = get_jac(dom, cod)

# Hessian tensor (dummy) in M_1, M_0*, M_0*
# (swapped with M_0*, M_1, M_0* for convinience)
hes = get_hes(dom, cod)

first_info(dom, cod, jac, hes)  # Just for print

# If composite 8 mappings
for i in range(1, 8):
    next_cod = random_dimension()           # Dimension of next codomain
    prev_info(i, cod, next_cod, jac, hes)   # Just for print

    pdv1 = get_jac(cod, next_cod)   # dTi/dxi
    pdv2 = get_hes(cod, next_cod)   # d2Ti/dxi2

    # Oneline code of the reccurence relations
    jac, hes = pdv1 @ jac, (pdv2 @ jac).T @ jac + pdv1 @ hes

    next_info(jac, hes) # Just for print
    cod = next_cod      # Rotation of codomain

print(jac.shape, hes.shape)

"""Output example
T0: M0 (6) -> M1 (2) jac: (6, 6) hes: (6, 6, 6) -> jac: (2, 6) hes: (6, 2, 6)
T1: M1 (2) -> M2 (4) jac: (2, 6) hes: (6, 2, 6) -> jac: (4, 6) hes: (6, 4, 6)
T2: M2 (4) -> M3 (5) jac: (4, 6) hes: (6, 4, 6) -> jac: (5, 6) hes: (6, 5, 6)
T3: M3 (5) -> M4 (2) jac: (5, 6) hes: (6, 5, 6) -> jac: (2, 6) hes: (6, 2, 6)
T4: M4 (2) -> M5 (5) jac: (2, 6) hes: (6, 2, 6) -> jac: (5, 6) hes: (6, 5, 6)
T5: M5 (5) -> M6 (2) jac: (5, 6) hes: (6, 5, 6) -> jac: (2, 6) hes: (6, 2, 6)
T6: M6 (2) -> M7 (3) jac: (2, 6) hes: (6, 2, 6) -> jac: (3, 6) hes: (6, 3, 6)
T7: M7 (3) -> M8 (1) jac: (3, 6) hes: (6, 3, 6) -> jac: (1, 6) hes: (6, 1, 6)
(1, 6) (6, 1, 6)
"""