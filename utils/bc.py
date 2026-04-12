"""Time-dependent boundary condition helpers."""

import dolfin as dl

class _TimeManager:
    """Helper class to manage time and trigger updates."""
    def __init__(self, bc_instance):
        self._bc = bc_instance
        self._t = 0.0

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, value):
        self._t = value
        self._bc.update(value)

class TimeDependentBoundaryCondition(dl.DirichletBC):
    """
    A class to define a time-dependent Dirichlet boundary condition.
    
    Parameters:
    -----------
    V : dolfin.FunctionSpace
        Function space to which the boundary condition applies.
    contact_position : float
        Position of the contact point where the boundary condition is applied.
    boundary : dolfin.SubDomain
        The subdomain defining the boundary where the condition is applied.
    """
    
    def __init__(self, Vh, loading_position, boundary):
        # Create a Constant that can be updated in-place
        self.value = dl.Constant(loading_position(0.0))
        # Initialize the parent class ONCE with this updatable Constant
        super().__init__(Vh, self.value, boundary)
        self.Vh = Vh
        self.boundary = boundary
        self.loading_position = loading_position
        self.function_arg = _TimeManager(self)

    def update(self, t):
        """
        Update the boundary condition based on the current time.
        
        Parameters:
        -----------
        t : float
            Current time in the simulation.
        """
        # Update the boundary condition based on the loading position and time
        new_value = self.loading_position(t)
        # This modifies the *existing* Constant object
        self.value.assign(dl.Constant(new_value))