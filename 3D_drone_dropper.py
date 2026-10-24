from dataclasses import dataclass, replace
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
import plotly.graph_objects as go
import pyvista as pv

#----------- Physical Constants --------------
R_AIR = 287.05         # specific gas constant of dry air, J/(kg*K)
P0 = 101325.0          # standard sea-level pressure, Pa
# ---------- Launch Conditions --------------- 
@dataclass(frozen = True)
class Params:
    speed: float = 15.0       # launch speed, m/s
    elevation_deg: float = 0.0  #xy angle
    azimuth_deg: float = 0.0   #xz angle 
    Cd: float | None = 0.3     #Default drag coefficient 
    A: float = 0.05         # cross-sectional area, m^2
    m: float = 4.0          # mass, kg
    wind_x: float = 0.0     # wind speed m/s in  x-dir (positive = tailwind)
    wind_y: float = 0.0     # wind speed m/s in  y-dir (positive = updraft)
    wind_z: float = 0.0     # wind speed m/s in  z-dir (positive = towards +z)
    g: float = 9.81         #gravity, m/s^2
    y0: float = 50.0         #release height m
    T: float = 288.15       #Temperature of sea level, Kelvin 

def initial_velocity(p: Params):
    elevation_theta = np.radians(p.elevation_deg)
    azimuth_theta = np.radians(p.azimuth_deg)
    vx_int, vy_int, vz_int = [p.speed * np.cos(elevation_theta) * np.cos(azimuth_theta), p.speed * np.sin(elevation_theta),
                              p.speed*np.sin(azimuth_theta) * np.cos(elevation_theta)]
    return vx_int, vy_int, vz_int
#------------Value Functions------------
mu = lambda T: 1.716e-5 * (T/273.15)**1.5 * (273.15 + 110.4) / (T + 110.4)

def cd_sphere(Re):
    """Morrison (2013) sphere drag correlation."""
    return (24/Re
            + 2.6*(Re/5) / (1 + (Re/5)**1.52)
            + 0.411*(Re/263000)**-7.94 / (1 + (Re/263000)**-8)
            + 0.25*(Re/1e6) / (1 + Re/1e6))
def air_density(y, p: Params):
    """Isothermal atmosphere at temperature p.T (ideal gas law + hydrostatic balance).
    rho0 = P0 / (R T) sets sea-level density; H = R T / g sets the scale height."""
    rho0 = P0 / (R_AIR * p.T)
    H = R_AIR * p.T / p.g
    return rho0 * np.exp(-y / H)


def drag_coefficient(p: Params, rho, v_rel):
    if p.Cd is not None:                 
        return p.Cd
    D = np.sqrt(4 * p.A / np.pi)         
    Re = max(rho * v_rel * D / mu(p.T), 1e-6)
    return cd_sphere(Re)
#------------ ODE Solver-----------------

def simulate(p: Params):
    vx0, vy0, vz0 = initial_velocity(p)
    initial_cond = [0.0, p.y0, 0.0, vx0, vy0, vz0]

    def flight_dynamics(t, s):
        x, y, z, vx, vy, vz = s
        rho = air_density(y, p)
        rvx, rvy, rvz = vx - p.wind_x, vy - p.wind_y, vz - p.wind_z
        v_rel = np.sqrt(rvx**2 + rvy**2 + rvz**2)
        k = 0.5 * rho * drag_coefficient(p, rho, v_rel) * p.A / p.m * v_rel
        return [vx, vy, vz, -k * rvx, -p.g - k * rvy, -k * rvz]
    
    def hit_ground(t, cond):
        return cond[1]

    hit_ground.terminal = True
    hit_ground.direction = -1
    sol = solve_ivp(flight_dynamics,(0, 300), y0=initial_cond,
            events=hit_ground, dense_output= True, rtol=1e-8, atol=1e-10)
    if sol.status !=  1:
        raise RuntimeError("Projectile never landed within the time frame")
    return sol
def solve_summarize(sol):
    return{"range": sol.y[0, -1], "max_height": sol.y[1].max(), "crossrange": sol.y[2,-1],"time": sol.t[-1], 
           "impact_speed": np.linalg.norm(sol.y[3:6, -1])}


# ------------ Vacuum Comparison ----------------
def vacuum_landing(p: Params):
    vx0, vy0, vz0 = initial_velocity(p)
    t = (vy0 + np.sqrt(vy0**2 + 2 * p.g * p.y0)) / p.g
    return vx0 * t, vz0 * t


def check_against_vacuum():
    p = replace(Params(Cd=0.0), elevation_deg=20.0, azimuth_deg=30.0)
    summary = solve_summarize(simulate(p))
    num_x, num_z = summary["range"], summary["crossrange"]
    ana_x, ana_z = vacuum_landing(p)

    ok_x = np.isclose(num_x, ana_x, rtol=1e-4)
    ok_z = np.isclose(num_z, ana_z, rtol=1e-4, atol=1e-6)
    if not (ok_x and ok_z):
        raise RuntimeError(
            f"Vacuum check FAILED: x numeric {num_x:.6f} vs analytic {ana_x:.6f} m, "
            f"z numeric {num_z:.6f} vs analytic {ana_z:.6f} m")
def check_planar_symmetry():
    p = replace(Params(), azimuth_deg = 0.0, elevation_deg = 35.0, wind_x = 10.0, wind_y = 10.0, wind_z = 0.0)
    sol = simulate(p)
    z_history = sol.y[2]
    worst_drift = np.max(np.abs(z_history))
    if  worst_drift > 1e-12:
        raise RuntimeError(f"Drift check FAILED:absolute largest drift was {worst_drift:.3e} m")
#------------ Wind Functions -------------------
def check_crosswind_sign():
    base = replace(Params(), elevation_deg = 0, azimuth_deg = 0, wind_x = 0, wind_y = 0)
    cross_range_pos = solve_summarize(simulate(replace(base, wind_z = 5.0)))["crossrange"]
    cross_range_neg = solve_summarize(simulate(replace(base, wind_z = -5.0)))["crossrange"]
    total = cross_range_pos + cross_range_neg
    if cross_range_pos <= 0:
        raise ValueError(f"+5 m/s crosswind gave crossrange {cross_range_pos:.3e} m, expected > 0")
    elif cross_range_neg >= 0:
        raise ValueError(f"-5 m/s crosswind gave crossrange {cross_range_neg:.3e} m, expected < 0")
    elif not np.isclose(total, 0, atol = 1e-9):
        raise ValueError(f"Mirror check FAILED: +5 gave {cross_range_pos:.6e} m, -5 gave {cross_range_neg:.6e} m, sum {total:.3e} m")
    

#-------------- Ballistic Checker for Drag Force -----------------
def ballistic_checker():
    base = Params(Cd = 0.3)
    scaled = replace(base, A = base.A*2, m = base.m*2)
    range1 = solve_summarize(simulate(base))["range"]
    range2 = solve_summarize(simulate(scaled))["range"]
    if not np.isclose(range1, range2, rtol = 1e-6):
        raise RuntimeError("The scaling between mass and area is incorrect")
    
def check_terminal_velocity_drop():
    p = Params(speed=0.0, elevation_deg=0.0, Cd=0.3, y0=50.0)
    T = simulate(p).t[-1]
    rho = air_density(p.y0/2, p)         # mid-drop density
    vt = np.sqrt(2 * p.m * p.g / (rho * p.Cd * p.A))
    fall = vt**2 / p.g * np.log(np.cosh(p.g * T / vt))  # analytic quadratic-drag fall distance
    if not np.isclose(fall, p.y0, rtol=1e-2):
        raise RuntimeError(f"Drop check FAILED: analytic fall {fall:.3f} m vs y0 {p.y0} m")

def check_density_temperature():
    rho_standard = air_density(0.0, Params(T=288.15))
    if not np.isclose(rho_standard, 1.225, rtol = 1e-3):
        raise RuntimeError(f"Sea level density check FAILED, got {rho_standard:.4f}, not ~1.225")
    rho_cold = air_density(0.0, Params(T = 240))
    rho_hot = air_density(0.0, Params(T = 330))
    if not rho_cold > rho_standard > rho_hot:
        raise RuntimeError("Density should decrease as Temperature increases")
#------------ Angle Optimization-----------------
def range_at_cond(angle_deg, base: Params):
    return solve_summarize(simulate(replace(base, elevation_deg=angle_deg)))["range"]

def angle_sweep(base: Params, angles):
    return np.array([range_at_cond(a, base) for a in angles])

def find_optimal_angle(base: Params):
    low, high = (10, 80)
    margin = 0.5
    res = minimize_scalar(lambda a: -range_at_cond(a, base),
                          bounds=(low, high), method="bounded")
    if res.x < low + margin or res.x > high - margin:
        raise ValueError(f"Best angle {res.x:.2f} deg is at the edge of ({low}, {high})")
    return res.x, -res.fun
#----------- Plotting ------------------
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]

def dashed_line(points, dash=3, gap=2):
    """Dashed polyline built as a single PolyData (no mesh merging)."""
    idx = np.arange(len(points) - 1)
    keep = idx[(idx % (dash + gap)) < dash]
    cells = np.column_stack([np.full(len(keep), 2), keep, keep + 1]).ravel()
    mesh = pv.PolyData(np.asarray(points, dtype=float))
    mesh.lines = cells
    return mesh

def build_scene(pl, base, traj_angles, show_vacuum, n):
    pl.set_background("white")
    legend = []
    for i, a in enumerate(traj_angles):
        c = COLORS[i % len(COLORS)]
        p = replace(base, elevation_deg=a)

        sol = simulate(p)
        t = np.linspace(0, sol.t[-1], n)
        x, y, z = sol.sol(t)[:3]
        pts = np.column_stack([x, -z, y])      # y (height) -> vertical axis
        pl.add_mesh(pv.lines_from_points(pts), color=c, line_width=3)
        pl.add_mesh(pv.PolyData(pts[-1:]), color=c, point_size=10,
                    render_points_as_spheres=True)
        legend.append([f"{a}°", c])

        if show_vacuum:                        # one vacuum curve per angle
            vx0, vy0, vz0 = initial_velocity(p)
            t_vac = (vy0 + np.sqrt(vy0**2 + 2 * p.g * p.y0)) / p.g
            tv = np.linspace(0, t_vac, n)
            vac = np.column_stack([vx0 * tv, -vz0 * tv,
                                   p.y0 + vy0 * tv - 0.5 * p.g * tv**2])
            pl.add_mesh(dashed_line(vac), color=c, line_width=2, opacity=0.6)

    if show_vacuum:
        legend.append(["vacuum (dashed)", "gray"])

    pl.show_bounds(xtitle="x downrange (m)", ytitle="z crossrange (m)",
                   ztitle="y height (m)", color="black")
    pl.add_legend(legend, bcolor="white", border=True)
    pl.add_title("3D trajectories (solid: drag + wind, dashed: vacuum)",
                 font_size=10, color="black")
    pl.view_isometric()

def plot_3d(base=None, traj_angles=(20, 30, 45, 60, 75), show_vacuum=True, n=100,
            save_path="projectile_sweep_3d.png", show=True, run_checks=True):
    if run_checks:
        check_against_vacuum()
        ballistic_checker()
        check_planar_symmetry()
        check_crosswind_sign()
        check_terminal_velocity_drop()
        check_density_temperature()

    if base is None:
        base = Params(azimuth_deg=20.0, wind_z=3.0)

    size = (1100, 800)

    if save_path:                              # reliable save, always off-screen
        pl = pv.Plotter(window_size=size, off_screen=True)
        build_scene(pl, base, traj_angles, show_vacuum, n)
        pl.screenshot(save_path)
        pl.close()

    if show:                                   # interactive native window
        pl = pv.Plotter(window_size=size)
        build_scene(pl, base, traj_angles, show_vacuum, n)
        target = save_path or "projectile_view.png"
        pl.add_key_event("s", lambda: pl.screenshot(target))  # saves current view
        pl.show()
   
if __name__ == "__main__":
        plot_3d()