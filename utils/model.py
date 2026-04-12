"""Base PDE model classes for time-dependent problems."""

import dolfin as dl
import ufl
import numpy as np
import hippylib as hp
from hippylib.modeling.variables import STATE, PARAMETER, ADJOINT

class LinearTimeDependentPDEVariationalProblem(hp.PDEProblem):
    def __init__(self, Vh, varf_handler, bc, bc0, u0, t_init, t_final):
        """
        varf_handler class
        conds = [u0, fwd_bc, adj_bc] : initial condition, (essential) fwd_bc, (essential) adj_bc
        When Vh[STATE] is MixedFunctionSpace, bc's are lists of DirichletBC classes 
        """

        self.Vh = Vh
        self.varf = varf_handler

        if isinstance(bc, dl.DirichletBC): 
            self.fwd_bc = [bc]
        else:
            self.fwd_bc = bc

        if isinstance(bc0, dl.DirichletBC):
            self.adj_bc = [bc0]
        else:
            self.adj_bc = bc0

        self.mesh = self.Vh[STATE].mesh()
        self.init_cond = u0
        self.t_init = t_init    
        self.t_final = t_final
        self.dt = varf_handler.dt
        self.times = np.arange(self.t_init, self.t_final+.5*self.dt, self.dt)

        self.linearize_x = None
        self.solverA = None
        self.solverAadj = None
        self._help_fwd = None
        self._help_adj = None

        M_varf = dl.inner(dl.TrialFunction(self.Vh[STATE]), dl.TestFunction(self.Vh[ADJOINT]))*dl.dx
        rhs_varf = dl.inner(dl.Function(self.Vh[STATE]), dl.TestFunction(self.Vh[ADJOINT]))*dl.dx
        self.M, dummy = dl.assemble_system(M_varf, rhs_varf, bcs=self.adj_bc)

    def setForwardSolver(self, m):
        
        if self._help_fwd is None:
            self._help_fwd = self.generate_parameter()
            reset = True
        else:
            help = m.copy()
            help.axpy(-1, self._help_fwd)
            reset = True if help.norm("l2") > 1e-13 else False
        if reset:
            u_old   = dl.Function(self.Vh[STATE])
            m_fun   = hp.vector2Function(m, self.Vh[PARAMETER])
            p       = dl.TestFunction(self.Vh[ADJOINT])
            du      = dl.TrialFunction(self.Vh[STATE])

            form = self.varf(du, u_old, m_fun, p, self.times[1])
            A_form = ufl.lhs(form)
            A, dummy = dl.assemble_system(A_form, ufl.rhs(form), bcs=self.adj_bc)
            if self.solverA is None: self.solverA = self._createLUSolver()
            self.solverA.set_operator(A)

            self._help_fwd.zero()
            self._help_fwd.axpy(1., m)
    
    def setAdjointSolver(self, m):

        if self._help_adj is None:
            self._help_adj = self.generate_parameter()
            reset = True
        else:
            help = m.copy()
            help.axpy(-1, self._help_adj)
            reset = True if help.norm("l2") > 1e-13 else False
        if reset:
            u       = dl.Function(self.Vh[STATE])
            u_old   = dl.Function(self.Vh[STATE])
            m_fun   = hp.vector2Function(m, self.Vh[PARAMETER])
            p       = dl.Function(self.Vh[ADJOINT])
            dp      = dl.TrialFunction(self.Vh[ADJOINT])
            du      = dl.TestFunction(self.Vh[STATE])

            form = self.varf(u, u_old, m_fun, p, self.times[1])
            Aadj_form = dl.derivative(dl.derivative(form, u, du), p, dp)
            rhs_varf = dl.inner(u, du)*dl.dx
            Aadj, dummy = dl.assemble_system(Aadj_form, rhs_varf, bcs=self.adj_bc)
            if self.solverAadj is None: self.solverAadj = self._createLUSolver()
            self.solverAadj.set_operator(Aadj)

            self._help_adj.zero()
            self._help_adj.axpy(1., m)

    def generate_vector(self, component = "ALL"):
        if component == "ALL":
            u = hp.TimeDependentVector(self.times)
            u.initialize(self.M, 1)
            a = dl.Function(self.Vh[PARAMETER]).vector()
            p = hp.TimeDependentVector(self.times)
            p.initialize(self.M, 0)
            return [u, a, p]
        elif component == STATE:
            u = hp.TimeDependentVector(self.times)
            u.initialize(self.M, 0)
            return u
        elif component == PARAMETER:
            return dl.Function(self.Vh[PARAMETER]).vector()
        elif component == ADJOINT:
            p = hp.TimeDependentVector(self.times)
            p.initialize(self.M, 0)
            return p
        else:
            raise Exception('Incorrect vector component')

    def generate_state(self):
        """ return a time dependent vector in the shape of the state """
        return self.generate_vector(component=STATE)

    def generate_parameter(self):
        """ return a time dependent vector in the shape of the adjoint """
        return self.generate_vector(component=PARAMETER)

    def generate_adjoint(self):
        """ return a time dependent vector in the shape of the adjoint """
        return self.generate_vector(component=ADJOINT)

    def generate_static_state(self):
        """ return a time dependent vector in the shape of the state """
        u = dl.Vector()
        self.M.init_vector(u, 1)
        return u 

    def generate_static_adjoint(self):
        """ return a static vector in the shape of the adjoint """
        p = dl.Vector()
        self.M.init_vector(p, 0)
        return p

    def init_parameter(self, a):
        """ initialize the parameter """
        dummy = self.generate_parameter()
        a.init( dummy.mpi_comm(), dummy.local_range() )
        
    def _set_time(self, bcs, t):
        """Update time on boundary conditions that expose a time manager."""
        for bc in bcs:
            try:
                bc.function_arg.t = t
            except:
                pass

    def solveFwd(self, out, x):
        """ Solve the possibly nonlinear time dependent Fwd Problem:
        Given a, find u such that
        \delta_p F(u,m,p;\hat_p) = 0 \for all \hat_p"""
        out.zero()

        u = dl.Function(self.Vh[STATE])
        u.vector().axpy(1., self.init_cond)
        out.store(u.vector(), self.t_init)

        m   = hp.vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        du  = dl.TrialFunction(self.Vh[STATE])
        dp  = dl.TestFunction(self.Vh[ADJOINT])

        b   = self.generate_static_state()

        self.setForwardSolver(x[PARAMETER])
        dummy = dl.PETScMatrix()
        
        for t in self.times[1:]:

            varf = self.varf(du, u, m, dp, t)
            A_form = ufl.lhs(varf)
            b_form = ufl.rhs(varf)
            self._set_time(self.fwd_bc, t)
            dl.assemble_system(A_form, b_form, bcs=self.fwd_bc, A_tensor=dummy, b_tensor=b)
            self.solverA.solve(u.vector(), b)

            out.store(u.vector(), t)

    def solveAdj(self, out, x, adj_rhs):
        """ Solve the linear time dependent Adj Problem: 
            Given a, u; find p such that
            \delta_u F(u,m,p;\hat_u) = 0 \for all \hat_u
        """
        out.zero()

        u       = dl.Function(self.Vh[STATE])
        u_old   = dl.Function(self.Vh[STATE])
        m       = hp.vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        du      = dl.TestFunction(self.Vh[STATE])
        p       = dl.Function(self.Vh[ADJOINT])

        b_adj   = self.generate_static_adjoint()
        rhs_t   = self.generate_static_state()

        self.setAdjointSolver(x[PARAMETER])

        for t in reversed(self.times[1:]):

            form = self.varf(u, u_old, m, p, t)
            b_adj_form = dl.Constant(-1.) * dl.derivative(form, u_old, du)
            dl.assemble(b_adj_form, tensor=b_adj)
            [bc.apply(b_adj) for bc in self.adj_bc]

            rhs_t.zero()
            adj_rhs.retrieve(rhs_t, t)
            b_adj.axpy(1., rhs_t)

            self.solverAadj.solve(p.vector(), b_adj)
            out.store(p.vector(), t)

    def evalGradientParameter(self, x, out):
        """Given u,m,p; eval \delta_m F(u,m,p; \hat_m) \for all \hat_m """
        out.zero()
        out_t = out.copy()

        dm      = dl.TestFunction(self.Vh[PARAMETER])
        u       = dl.Function(self.Vh[STATE])
        p       = dl.Function(self.Vh[ADJOINT])
        u_old   = dl.Function(self.Vh[STATE])
        m       = hp.vector2Function(x[PARAMETER], self.Vh[PARAMETER])

        x[STATE].retrieve(u_old.vector(), self.times[0])

        for t in self.times[1:]:
            x[STATE].retrieve(u.vector(), t)
            x[ADJOINT].retrieve(p.vector(), t)
            form = self.varf(u, u_old, m, p, t)
            out_t.zero()
            dl.assemble(dl.derivative(form, m, dm), tensor=out_t)
            out.axpy(1., out_t)

    def setLinearizationPoint(self, x, gauss_newton_approx=False):
        """ Set the values of the state and parameter
            for the incremental Fwd and Adj solvers """
        self.linearize_x = x
        self.gauss_newton_approx = gauss_newton_approx
        self.setForwardSolver(x[PARAMETER])
        self.setAdjointSolver(x[PARAMETER])     

    def _solveIncrementalFwd(self, out, rhs):
        out.zero()
        u       = dl.Function(self.Vh[STATE])
        u_old   = dl.Function(self.Vh[STATE])
        m       = hp.vector2Function(self.linearize_x[PARAMETER], self.Vh[PARAMETER]) 
        dp      = dl.TestFunction(self.Vh[ADJOINT])
        uhat    = dl.Function(self.Vh[STATE])
        
        b_inc   = self.generate_static_state()

        self.linearize_x[STATE].retrieve(u_old.vector(), self.times[0])

        for t in self.times[1:]:

            form = self.varf(u, u_old, m, dp, t)
            binc_form = dl.Constant(-1.)*dl.derivative(form, u_old, uhat)
            dl.assemble(binc_form, tensor=b_inc)
            [bc.apply(b_inc) for bc in self.adj_bc]
            b_inc.axpy(1., rhs.view(t))

            self.solverA.solve(uhat.vector(), b_inc)
            out.store(uhat.vector(), t)

            self.linearize_x[STATE].retrieve(u_old.vector(), t)

  
    def _solveIncrementalAdj(self, out, rhs):

        out.zero()

        u       = dl.Function(self.Vh[STATE])
        u_old   = dl.Function(self.Vh[STATE])
        m       = hp.vector2Function(self.linearize_x[PARAMETER], self.Vh[PARAMETER])
        du_old  = dl.TestFunction(self.Vh[STATE])
        phat    = dl.Function(self.Vh[ADJOINT])

        b_adj_inc = self.generate_static_adjoint()

        for it, t in enumerate(reversed(self.times[1:])):

            self.linearize_x[STATE].retrieve(u_old.vector(), self.times[it-1])

            form = self.varf(u, u_old, m, phat, t)
            b_adj_form  = dl.Constant(-1.)*dl.derivative(form, u_old, du_old)
            dl.assemble(b_adj_form, tensor=b_adj_inc)
            [bc.apply(b_adj_inc) for bc in self.adj_bc]
            b_adj_inc.axpy(1., rhs.view(t))

            self.solverAadj.solve(phat.vector(), b_adj_inc)
            out.store(phat.vector(), t)


    def solveIncremental(self, out, rhs, is_adj):
        """ If is_adj = False:
            Solve the forward incremental system:
            Given u, a, find \tilde_u s.t.:
            \delta_{pu} F(u,a,p; \hat_p, \tilde_u) = rhs for all \hat_p.
            
            If is_adj = True:
            Solve the adj incremental system:
            Given u, a, find \tilde_p s.t.:
            \delta_{up} F(u,a,p; \hat_u, \tilde_p) = rhs for all \delta_u.
        """
        if is_adj:
            return self._solveIncrementalAdj(out, rhs)
        else:
            return self._solveIncrementalFwd(out, rhs)


    def applyC(self, dm, out):
        out.zero()

        out_t = self.generate_static_adjoint() 
        u = dl.Function(self.Vh[STATE])
        u_old = dl.Function(self.Vh[STATE])
        m = hp.vector2Function(self.linearize_x[PARAMETER], self.Vh[PARAMETER])
        p = dl.Function(self.Vh[ADJOINT])
        
        dp = dl.TestFunction(self.Vh[ADJOINT])
        
        dm_fun = hp.vector2Function(dm, self.Vh[PARAMETER])

        self.linearize_x[STATE].retrieve(u_old.vector(), self.times[0])
        for t in self.times[1:]:
            self.linearize_x[STATE].retrieve(u.vector(), t)
            self.linearize_x[ADJOINT].retrieve(p.vector(), t)
            form = self.varf(u, u_old, m, p, t)
            cvarf = dl.derivative(dl.derivative(form, p, dp), m, dm_fun)
            out_t.zero()
            dl.assemble(cvarf, tensor=out_t)
            [bc.apply(out_t) for bc in self.adj_bc]

            self.linearize_x[STATE].retrieve(u_old.vector(), t)
            out.store(out_t, t)

    def applyCt(self, dp, out):
        out.zero()
        out_t = self.generate_parameter()
        
        u = dl.Function(self.Vh[STATE])
        u_old = dl.Function(self.Vh[STATE])
        m = hp.vector2Function(self.linearize_x[PARAMETER], self.Vh[PARAMETER])
        p = dl.Function(self.Vh[ADJOINT])
        
        dm = dl.TestFunction(self.Vh[PARAMETER])
        
        dp_fun = dl.Function(self.Vh[ADJOINT])

        self.linearize_x[STATE].retrieve(u_old.vector(), self.times[0])
        for t in self.times[1:]:
            self.linearize_x[STATE].retrieve(u.vector(), t)
            self.linearize_x[ADJOINT].retrieve(p.vector(), t)
            dp.retrieve(dp_fun.vector(), t)
            form = self.varf(u, u_old, m, p, t)
            cvarf_adj = dl.derivative(dl.derivative(form, p, dp_fun), m, dm)
            out_t.zero()
            dl.assemble(cvarf_adj, tensor=out_t)

            self.linearize_x[STATE].retrieve(u_old.vector(), t)
            
            out.axpy(1., out_t)


    def applyWuu(self, du, out):

        out.zero()
        
        if self.gauss_newton_approx:
            return
        
        u     = dl.Function(self.Vh[STATE])
        u_old = dl.Function(self.Vh[STATE])
        m = hp.vector2Function(self.linearize_x[PARAMETER], self.Vh[PARAMETER])
        p = dl.Function(self.Vh[ADJOINT])

        du_fun  = dl.Function(self.Vh[STATE])
        du_old  = dl.Function(self.Vh[STATE])
        
        du_test  = dl.TestFunction(self.Vh[STATE])
        du_old_test  = dl.TestFunction(self.Vh[STATE])

        self.linearize_x[STATE].retrieve(u_old.vector(), self.times[0])
        du.retrieve(du_old.vector(), self.times[0])
        for t in self.times[1:]:
            self.linearize_x[STATE].retrieve(u.vector(), t)
            self.linearize_x[ADJOINT].retrieve(p.vector(), t)
            
            du.retrieve(du_fun.vector(), t)
            
            form  = self.varf(u, u_old, m, p, t)
            varf = dl.derivative(dl.derivative(form, u, du_fun), u, du_test) + \
                   dl.derivative(dl.derivative(form, u_old, du_old), u_old, du_old_test)
                   
            out_t = dl.assemble(varf)
            [bc.apply(out_t) for bc in self.adj_bc]
            
            self.linearize_x[STATE].retrieve(u_old.vector(), t)
            du.retrieve(du_old.vector(), t)

            out.store(out_t, t)


    def applyWum(self, dm, out):
        out.zero()
        
        if self.gauss_newton_approx:
            return
        
        u = dl.Function(self.Vh[STATE])
        u_old  = dl.Function(self.Vh[STATE])
        m = hp.vector2Function(self.linearize_x[PARAMETER], self.Vh[PARAMETER])
        p = dl.Function(self.Vh[ADJOINT])
        
        dm_fun = hp.vector2Function(dm, self.Vh[PARAMETER]) 
        
        du_test  = dl.TestFunction(self.Vh[STATE])
        du_old_test  = dl.TestFunction(self.Vh[STATE])
        
        self.linearize_x[STATE].retrieve(u_old.vector(), self.times[0])    

        for t in self.times[1:]:
            self.linearize_x[STATE].retrieve(u.vector(), t)
            self.linearize_x[ADJOINT].retrieve(p.vector(), t)

            form  = self.varf(u, u_old, m, p, t)
            varf = dl.derivative(dl.derivative(form, m, dm_fun), u, du_test) + \
                   dl.derivative(dl.derivative(form, m, dm_fun), u_old, du_old_test)
                   
            out_t = dl.assemble(varf)
            [bc.apply(out_t) for bc in self.adj_bc]

            self.linearize_x[STATE].retrieve(u_old.vector(), t)   
            out.store(out_t, t)


    def applyWmu(self, du, out):
        out.zero()
        
        if self.gauss_newton_approx:
            return
        
        u = dl.Function(self.Vh[STATE])
        u_old = dl.Function(self.Vh[STATE])
        m = hp.vector2Function(self.linearize_x[PARAMETER], self.Vh[PARAMETER])
        p = dl.Function(self.Vh[ADJOINT])
        
        du_fun  = dl.Function(self.Vh[STATE])
        du_old  = dl.Function(self.Vh[STATE])
        
        dm_test = dl.TestFunction(self.Vh[PARAMETER])

        self.linearize_x[STATE].retrieve(u_old.vector(), self.times[0])
        du.retrieve(du_old.vector(), self.times[0])
        
        for t in self.times[1:]:
            self.linearize_x[STATE].retrieve(u.vector(), t)
            self.linearize_x[ADJOINT].retrieve(p.vector(), t)
            
            du.retrieve(du_fun.vector(), t)
            
            form  = self.varf(u, u_old, m, p, t)
            varf = dl.derivative(dl.derivative(form, u, du_fun), m, dm_test) + \
                   dl.derivative(dl.derivative(form, u_old, du_old), m, dm_test)
                   
            out_t = dl.assemble(varf)
            
            self.linearize_x[STATE].retrieve(u_old.vector(), t)
            du.retrieve(du_old.vector(), t)

            out.axpy(1., out_t)


    def applyWmm(self, dm, out):
        out.zero()
        
        if self.gauss_newton_approx:
            return

        out_t = self.generate_parameter()
        u = dl.Function(self.Vh[STATE])
        u_old = dl.Function(self.Vh[STATE])
        m = hp.vector2Function(self.linearize_x[PARAMETER], self.Vh[PARAMETER])
        p = dl.Function(self.Vh[ADJOINT])   
        
        dm_fun = hp.vector2Function( dm, self.Vh[PARAMETER])
        
        dm_test = dl.TestFunction(self.Vh[PARAMETER])

        self.linearize_x[STATE].retrieve(u_old.vector(), self.times[0])
        for t in self.times[1:]:
            self.linearize_x[STATE].retrieve(u.vector(), t)
            self.linearize_x[ADJOINT].retrieve(p.vector(), t)

            form = self.varf(u, u_old, m, p, t)
            varf = dl.derivative(dl.derivative(form, m, dm_fun), m, dm_test)
            out_t.zero()
            dl.assemble(varf, tensor=out_t)

            self.linearize_x[STATE].retrieve(u_old.vector(), t)
            out.axpy(1., out_t)


    def apply_ij(self,i,j, dir, out): 
        """
            Given u, a, p; compute 
            \delta_{ij} F(u,a,p; \hat_i, \tilde_j) in the direction \tilde_j = dir for all \hat_i
        """
        KKT = {}
        KKT[STATE, STATE] = self.applyWuu
        KKT[PARAMETER, STATE] = self.applyWmu
        KKT[STATE, PARAMETER] = self.applyWum
        KKT[PARAMETER, PARAMETER] = self.applyWmm
        KKT[ADJOINT, STATE] = None 
        KKT[STATE, ADJOINT] = None 

        KKT[ADJOINT, PARAMETER] = self.applyC
        KKT[PARAMETER, ADJOINT] = self.applyCt
        KKT[i,j](dir, out)
        
    def exportState(self, u, fname, component=None):
        ufun = dl.Function(self.Vh[STATE], name="state")
        with  dl.XDMFFile(fname) as fid:
            fid.parameters["functions_share_mesh"] = True
            fid.parameters["rewrite_function_mesh"] = False
            for t in self.times:
                u.retrieve(ufun.vector(), t)
                if component is None:
                    fid.write(ufun, t)
                else:
                    fid.write(ufun.sub(component), t)

    def _createLUSolver(self):
        return hp.PETScLUSolver(self.Vh[STATE].mesh().mpi_comm())