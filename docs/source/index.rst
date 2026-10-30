:html_theme.sidebar_secondary.remove: true

######################
mappy-ds documentation
######################

**Version**: |version|

**mappy-ds** computes Poincaré maps of smooth and non-smooth (hybrid)
dynamical systems — ODEs with impacts, resets, switching or state jumps —
together with their exact Jacobian matrices and Hessian tensors. A system is
described as a set of smooth *modes* (continuous flows that stop on a border,
and discrete maps) connected by transition rules; following the mode concept
of Miino (NOLTA, 2024), the package composes them into a single
diffeomorphism. On top of that it finds periodic points, locates tangent,
period-doubling and Neimark–Sacker bifurcations, and continues both along
parameters.

.. code-block:: sh

   pip install mappy-ds

.. note::

   The distribution is called ``mappy-ds`` and the import name is
   ``mappy_ds``. The PyPI project named ``mappy`` is an unrelated package
   (Python bindings of the minimap2 aligner).

.. toctree::
   :hidden:
   :maxdepth: 1

   contents/getting_started
   contents/user_guide
   contents/api-reference/index

.. grid:: 1 1 3 3
   :gutter: 2

   .. grid-item-card::
      :img-top: _static/icons/rocket-takeoff.svg

      Getting started
      ^^^^^^^^^^^^^^^

      Install the package and build, evaluate and analyse your first
      Poincaré map — the Izhikevich neuron — in five minutes.

      .. button-ref:: contents/getting_started
         :expand:
         :color: secondary
         :click-parent:

         To the tutorial

   .. grid-item-card::
      :img-top: _static/icons/book-half.svg

      User guide
      ^^^^^^^^^^

      The mode concept, how derivatives are computed, periodic points,
      bifurcations and continuation, plotting, and worked examples.

      .. button-ref:: contents/user_guide
         :expand:
         :color: secondary
         :click-parent:

         To the user guide

   .. grid-item-card::
      :img-top: _static/icons/code-slash.svg

      API reference
      ^^^^^^^^^^^^^

      Detailed description of every public function and class, its
      parameters and its return values.

      .. button-ref:: contents/api-reference/index
         :expand:
         :color: secondary
         :click-parent:

         To the API reference

Reference
=========

Y. Miino, "Mode concept to construct diffeomorphisms representing
smooth/non-smooth dynamical systems", *Nonlinear Theory and Its Applications,
IEICE*, vol. 15, no. 2, pp. 284–298, 2024.
`doi:10.1587/nolta.15.284 <https://doi.org/10.1587/nolta.15.284>`_
