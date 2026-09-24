

class ProgramParameters:
    """All parameters for the simulation program."""

    def __init__(self, args, name):
        """Initialize parameters for the simulation with input arguments."""
        # # read configuration file
        # with open(args.config) as yamlfile:
        #     self.cfg = yaml.safe_load(yamlfile)
        
        # path variables
        self.scene_path = args.scene
        # self.config_path = args.config
        self.output_path = args.output
        self.image_path = args.output + "/mjc_image_%03d.png"
        self.program_path = name

        # simulation
        self.solver_type     = args.solver_type     if (args.solver_type >= 0)     else 1  # default solver = CG
        self.integrator_type = args.integrator_type if (args.integrator_type >= 0) else 0  # default integrator = Euler
        self.flag_draw_contacts = False
        self.flag_compute_pressure = False

        # rendering
        self.flag_save_images = args.save_images
        self.flag_headless = args.headless
        self.fps = 24

        # scene
        self.flag_endless = False
        self.total_time = 5.0
        self.max_runtime = 1200.0
        self.asset_root_path = ".."
        self.default_pose = [0.0, 0.0, 0.0]

        # statistics
        self.num_tetrahedrals = 0
        self.num_triangles = 0
        self.output_file = open(self.output_path+"/mjc_output.txt", 'w')
        self.output_file.write("MuJoCo, "+self.program_path+": "+str(self.scene_path)+"\n")
        self.particle_state_tensor = []
        self.dt_save = 0.01


integratorNames = [
    "mjINT_EULER",          # semi-implicit Euler
    "mjINT_RK4",            # 4th-order Runge Kutta
    "mjINT_IMPLICIT",       # implicit in velocity
    "mjINT_IMPLICITFAST"]   # implicit in velocity, no rne derivative

solverNames = [
    "mjSOL_PGS",            # PGS    (dual)
    "mjSOL_CG",             # CG     (primal)
    "mjSOL_NEWTON"]         # Newton (primal)


import os

def make_output_line(len=32):
    return "|"+len*"-"+"|\n"

def make_data(name, val):
    return "| {:<30} | {}\n".format(name, val)

def print_path_info(os, params):
    os.write("|---Paths"+24*"-"+"|\n")
    os.write(make_data("Scene File",params.scene_path))
    # os.write(make_data("Configuration File",params.config_path))
    os.write(make_data("Result Output File",params.output_path))
    os.write(make_data("Image Files",params.image_path))

def print_result_info(os, params, t, step_counter, t_all, t_step, t_view, t_render):
    os.write("|---Properties"+19*"-"+"|\n")
    os.write(make_data("Number of Tetrahedrals",params.num_tetrahedrals))
    os.write(make_data("Number of Triangles",params.num_triangles))
    os.write("|---Simulation"+19*"-"+"|\n")
    os.write(make_data("Number of Simulation-Step",step_counter))
    os.write(make_data("Total Simulation-Time",t))
    os.write(make_data("Avg. Simulation-Time",t/step_counter))
    os.write("|---Performance"+18*"-"+"|\n")
    os.write(make_data("Total Time (Wall)",  t_all.get_wall_time()))
    os.write(make_data(" Avg. Time (Wall)",  t_all.get_wall_time()/step_counter))
    os.write(make_data("Total Time (CPU)",   t_all.get_cpu_time()))
    os.write(make_data(" Avg. Time (CPU)",   t_all.get_cpu_time()/step_counter))
    os.write(make_data("Simulation Time (Wall)",       t_step.get_wall_time()))
    os.write(make_data(" Avg. Simulation Time (Wall)", t_step.get_wall_time()/step_counter))
    os.write(make_data("Simulation Time (CPU)",        t_step.get_cpu_time()))
    os.write(make_data(" Avg. Simulation Time (CPU)",  t_step.get_cpu_time()/step_counter))
    os.write(make_data("Viewer Time (Wall)",       t_view.get_wall_time()))
    os.write(make_data(" Avg. Viewer Time (Wall)", t_view.get_wall_time()/step_counter))
    os.write(make_data("Viewer Time (CPU)",        t_view.get_cpu_time()))
    os.write(make_data(" Avg. Viewer Time (CPU)",  t_view.get_cpu_time()/step_counter))
    os.write(make_data("Rendering Time (Wall)",       t_render.get_wall_time()))
    os.write(make_data(" Avg. Rendering Time (Wall)", t_render.get_wall_time()/step_counter))
    os.write(make_data("Rendering Time (CPU)",        t_render.get_cpu_time()))
    os.write(make_data(" Avg. Rendering Time (CPU)",  t_render.get_cpu_time()/step_counter))

def print_sim_params(os, params):
    os.write("## Simulation Parameters\n")
    os.write(" solver type {} \n".format(params.solver_type))
    os.write(" integrator type {} \n".format(params.integrator_type))

def print_model_info(os, m):
    os.write("# Model information:\n")
    os.write(" Number of generalized coordinates (nq = dim(qpos)): {} \n".format(m.nq))
    os.write(" Number of degrees of freedom (nv = dim(qvel)): {} \n".format(m.nv))
    os.write(" Number of actuators/controls (nu = dim(ctrl)): {} \n".format(m.nu))
    os.write(" Number of bodies  (nbody): {} \n".format(m.nbody))
    os.write(" Number of geoms   (ngeom): {} \n".format(m.ngeom))
    os.write(" Number of flexes  (nflex): {} \n".format(m.nflex))
    os.write(" Number of meshes  (nmesh): {} \n".format(m.nmesh))
    os.write(" Number of textures (ntex): {} \n".format(m.ntex) )
    os.write(" Number of sensors (nsensor): {} \n".format(m.nsensor))
    os.write(" Number of cameras  (ncam): {} \n".format(m.ncam))

def print_flex_info(os, m):
    if (m.nflex > 0):
        os.write("# Flex information:\n")
        os.write(" Number of flexes (nflex): {} \n".format(m.nflex) )
        os.write(" Number of elements in all flexes (nflexelem): {} \n".format(m.nflexelem) )
        os.write(" Number of vertices in all flexes (nflexvert): {} \n".format(m.nflexvert) )
        os.write(" Number of edges in all flexes    (nflexedge): {} \n".format(m.nflexedge) )
        os.write(" Number of element vertex ids in all flexes (nflexelemdata): {} \n".format(m.nflexelemdata) )

        for i in range(0, m.nflex):
            os.write(" ## Flex ("+str(i)+") \n" )
            os.write("  Flex dimension (flex_dim): {} \n".format(m.flex_dim[i]) )
            os.write("  Number of vertices (flex_vertnum): {} \n".format(m.flex_vertnum[i]) )
            os.write("  Number of edges    (flex_edgenum): {} \n".format(m.flex_edgenum[i]) )
            os.write("  Number of elements (flex_elemnum): {} \n".format(m.flex_elemnum[i]) )
            os.write("  Number of shells  (flex_shellnum): {} \n".format(m.flex_shellnum[i]) )
            os.write("  Edge stiffness (flex_edgestiffness): {}  \n".format(m.flex_edgestiffness[i]) )
            os.write("  First element address (flex_elemadr): {} \n".format(m.flex_elemadr[i]) )
            os.write("  First vertex address  (flex_vertadr): {} \n".format(m.flex_vertadr[i]) )

    else:
        os.write(" # No Flex-objects.\n")


def print_simulation_info(os, m):
    os.write("# Simulation information:\n" )
    os.write(" Integration mode, mjtIntegrator (opt.integrator): {} - ".format(m.opt.integrator) +str(integratorNames[m.opt.integrator]) + "\n" )
    os.write(" Solver algorithm, mjtSolver     (opt.solver):     {} - ".format(m.opt.solver)     +str(solverNames[m.opt.solver]) + "\n" )


import time

class Timer:
    """Helper class for timing."""

    def __init__(self):
        self.t_wall_begin = 0
        self.t_wall_end = 0
        self.t_wall_elapsed = 0.0
        self.t_cpu_begin = 0
        self.t_cpu_end = 0
        self.t_cpu_elapsed = 0.0

    def start_clock(self):
        self.t_wall_begin = time.time()
        self.t_cpu_begin = time.process_time()
    
    def end_clock(self):
        self.t_cpu_end = time.process_time()
        self.t_wall_end = time.time()
    
    # def add_time(self):
        self.t_wall_elapsed += self.t_wall_end - self.t_wall_begin
        self.t_cpu_elapsed += self.t_cpu_end - self.t_cpu_begin

    def get_wall_time(self):
        return self.t_wall_elapsed

    def get_cpu_time(self):
        return self.t_cpu_elapsed


