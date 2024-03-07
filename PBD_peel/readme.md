## Grape Peeling Demo

### Boundary constraints

Boundary constraints are formulated as follow:

- Adding a virtual inifinite-mass particle at every mesh particle position at resting (init) state
- Formulate a distance constraint between a mesh particle and its corresponding virtual particle. The distance constraint has resting length of zero
- All such constraints are independent, and can be solved all in parallel
- Note that position of the virtual particle never changed through out the simulation (can be changed in the future)
- It makes the mesh seemed to be softly attached to its initial position

### How peeling work

When the energy of a certain boundary constraint exceed a limit, we consider this connection is broken. The stiffness is then set to be very small.

### Demo

To run the demo, simply do `python peel_grape.py`

In the demo, a grasping point follows a predefined trajectory. A thin-shell layer is then "peeled off". 

Currently, the grape is merely a background for visualization. It is not a part of the simulation
