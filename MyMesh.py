from fenics import *
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

def vertex_normal(mesh, boundary_parts, i):
    """Calculates the normal of a given vertex i of the mesh."""
    
    ver = Vertex(mesh, i) #vertex i of the mesh
    n = Point(0.0, 0.0) #initial normal
    div = 0.0 #number of normal facets
    
    for fac in entities(ver, mesh.geometry().dim()-1):
    
        #for each facet next to the vertex i
        f = Facet(mesh, fac.index())
    
        if f.exterior()==True:
            #if it touches the border
            #we use its normal
            n+=f.normal()
            div+=1
    
    n/=div #normalisation
    
    return n


def compute_normal_field(omega, mf):
    
    #Boundary Mesh
    gamma = BoundaryMesh(omega, "exterior")
    mapa = gamma.entity_map(0)
    
    # normal_field is a vectorial function of the boundary space (a vector field defined on the boundary)
    N = VectorFunctionSpace(gamma, "CG", 1)
    normal_field = Function(N)
    
    #its components are functions of the boundary space
    V = FunctionSpace(gamma, "CG", 1)
    normal_x = Function(V)
    normal_y = Function(V)
    
    for ver in entities(gamma, 0):
        
        #for every vertex of the Boundary Mesh
        #we get its normal
        i = mapa[ver.index()] #index of the vertex in the mesh omega
        point = vertex_normal(omega, mf, i) #normal of the vertex in omega
        
        #normal for the integral is the opposite to the one we have computed
        normal_x.vector()[ver.index()] = -1.0*point[0]
        normal_y.vector()[ver.index()] = -1.0*point[1]
    
    #x and y are just seen as TestFunction
    #a = inner(normal_field[0], x)*dx + inner(normal_field[1], y)*dx is bilinear
    #L = inner(normal_x, x)*dx + inner(normal_y, y)*dx is linear
    (x, y) = TestFunctions(N)
    solve(inner(normal_field[0], x)*dx + inner(normal_field[1], y)*dx - inner(normal_x, x)*dx -inner(normal_y, y)*dx == 0, normal_field)    
    
	#We have computed normal_field on the boundary mesh
    #Dirichlet Boundary must be initialized by Function in Vector Space over Omega (only on Gamma fails)
    #normal_fieldV is a vector function on the global space
    V_vec = VectorFunctionSpace(omega, "CG", 1)
    normal_fieldV = Function(V_vec)
    
    for ver in entities(gamma, 0):
        
        #for each vertex of the boundary
        #normal_fieldV equals normal_field on the boundary
        i = mapa[ver.index()]#index of the vertex in the mesh omega
        normal_fieldV.vector()[i] = normal_field.vector()[ver.index()] 
        normal_fieldV.vector()[i+omega.num_vertices()] = normal_field.vector()[ver.index()+gamma.num_vertices()]
    
    # We are solving a vector function on the whole mesh
    deform = TrialFunction(V_vec)
    
    # TestFunction
    v = TestFunction(V_vec)
    
    #Heat problem to solve with nul initial condition (to try to smooth the normal_field in the mesh) 
    a = 0.01*inner(nabla_grad(deform), nabla_grad(v))*dx + 0.01*inner(deform,v)*ds(4)
    L = inner(Constant((0.0,0.0)),v)*dx
    
    #bc1 : on the obstacle (circle), vector field we compute V_vec is equal to normal_fieldV
    bc1 = DirichletBC(V_vec, normal_fieldV, mf, 15)     #Obstacle
    bc2 = DirichletBC(V_vec, Constant((0,0)), mf, 13)    #outflow
    bc3 = DirichletBC(V_vec, Constant((0,0)), mf,14)    #noslip
    bc4 = DirichletBC(V_vec, Constant((0,0)), mf, 12)    #inflow
    bc = [bc1, bc2, bc3, bc4]
    
    #solving
    deform = Function(V_vec)
    solve(a==L, deform, bcs=bc)
    return deform

def grad2n(nx, ny, u, un):
    """returns the functions representing the x and y component of un = <grad(u),grad(u)>.n"""
    #Defining the function space
    Vn = u.function_space()
    mesh = Vn.mesh()
    Wn = FunctionSpace(mesh, 'CG', 1)
    
    #get the x, y components of <grad(u),grad(u)>.n in Vn
    grad_nu_x =inner(un, nx)
    grad_nu_y = inner(un, ny)
    
    #x, y components of <grad(u),grad(u)>.n in Wn
    grad_nu_x = project(grad_nu_x, Wn)
    grad_nu_y = project(grad_nu_y, Wn)

    gnux = Function(Wn, grad_nu_x.vector())
    gnuy = Function(Wn, grad_nu_y.vector())
    return gnux, gnuy

def mesh_surface(mesh):
	"""Calculates surface of a mesh."""
	
	Wu = FunctionSpace(mesh, 'CG', 1)
	e = Expression('1.0', degree=1)
	#function equal to 1 on each vertex of the mesh
	unity = interpolate(e, Wu)
	surface = unity*dx
	
	#its integral is the surface of the mesh
	surface = assemble(surface)
	
	return surface