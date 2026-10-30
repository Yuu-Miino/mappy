import numpy as np

Hg = np.random.randint(100, size=18).reshape(2, 3, 3)   # X, W*, W*
Hg = (Hg + Hg.transpose(0, 2, 1)) / 2                   # Make Hg symmetric
Jf = np.random.randint(100, size=12).reshape(3, 4)      # W, V*

# (2, 4) contraction (for W*, W)
cont24 = np.trace(np.tensordot(Hg, Jf, axes=0), axis1=1, axis2=3)

# (3, 4) contraction (for W*, W)
cont34 = np.trace(np.tensordot(Hg, Jf, axes=0), axis1=2, axis2=3)

# @ operation
atop = Hg @ Jf

# Comparison
print(cont24 == atop) # (2, 3, 4) tensor in X, W*, V*, with All True
print(cont34 == atop) # (2, 3, 4) tensor in X, W*, V*, with All True

# (2, 4) contraction (for W*, W)
cont24 = np.trace(np.tensordot(Hg @ Jf, Jf, axes=0), axis1=1, axis2=3)

# @ operation
atop = (Hg @ Jf).transpose(0, 2, 1) @ Jf

print(cont24 == atop) # (2, 4, 4) tensor in X, V*, V*, with All True

Jg = np.random.randint(100, size=6).reshape(2, 3)       # X, W*
Hf = np.random.randint(100, size=48).reshape(3, 4, 4)   # W, V*, V*
Hf = (Hf + Hf.transpose(0, 2, 1)) / 2                   # Make Hf symmetric

# (2, 3) contraction (for W*, W)
cont23 = np.trace(np.tensordot(Jg, Hf, axes=0), axis1=1, axis2=2)

# @ operation
atop = Jg @ Hf.transpose(1, 0, 2)   # (4, 2, 4) tensor
atop = atop.transpose(1, 0, 2)      # (2, 4, 4) tensor

print(cont23 == atop) # (2, 4, 4) tensor in X, V*, V*, with All True

# Einstein notation
ein = np.einsum('ij, jkl->ikl', Jg, Hf) # X W*, W V* V* -> X V* V*

print(cont23 == ein)