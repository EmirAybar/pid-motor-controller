import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────
#  PID Controller Class
# ─────────────────────────────────────────

class PIDController:
    """
    Discrete-time PID Controller with anti-windup.

    Parameters
    ----------
    Kp : float  — Proportional gain
    Ki : float  — Integral gain
    Kd : float  — Derivative gain
    dt : float  — Time step (seconds)
    output_limit : tuple — (min, max) output saturation
    """

    def __init__(self, Kp, Ki, Kd, dt, output_limit=(-100, 100)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.output_min, self.output_max = output_limit

        self._integral = 0.0
        self._prev_error = 0.0

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0

    def compute(self, setpoint, measurement):
        error = setpoint - measurement

        # Proportional term
        P = self.Kp * error

        # Integral term with anti-windup clamping
        self._integral += error * self.dt
        I = self.Ki * self._integral

        # Derivative term (based on error change)
        derivative = (error - self._prev_error) / self.dt
        D = self.Kd * derivative
        self._prev_error = error

        # Total output with saturation
        output = P + I + D
        output = np.clip(output, self.output_min, self.output_max)

        return output, P, I, D


# ─────────────────────────────────────────
#  DC Motor Model (First-Order System)
# ─────────────────────────────────────────

class DCMotor:
    """
    Simplified first-order DC motor model.
    
    Transfer function: G(s) = K / (tau*s + 1)
    Discretized with Euler method.

    Parameters
    ----------
    K   : float — Motor gain (RPM per unit input)
    tau : float — Time constant (seconds)
    dt  : float — Time step
    """

    def __init__(self, K=10.0, tau=0.5, dt=0.01):
        self.K = K
        self.tau = tau
        self.dt = dt
        self.speed = 0.0  # Initial speed (RPM)

    def reset(self):
        self.speed = 0.0

    def update(self, control_input):
        # Euler discretization: speed[n+1] = speed[n] + dt/tau * (K*u - speed[n])
        d_speed = (self.dt / self.tau) * (self.K * control_input - self.speed)
        self.speed += d_speed
        return self.speed


# ─────────────────────────────────────────
#  Simulation Function
# ─────────────────────────────────────────

def run_simulation(Kp, Ki, Kd, setpoint_profile, t, label, color):
    dt = t[1] - t[0]
    pid = PIDController(Kp, Ki, Kd, dt, output_limit=(-100, 100))
    motor = DCMotor(K=10.0, tau=0.5, dt=dt)

    speeds, outputs, P_terms, I_terms, D_terms, errors = [], [], [], [], [], []

    for i, ti in enumerate(t):
        sp = setpoint_profile[i]
        speed = motor.speed
        control, P, I, D = pid.compute(sp, speed)
        motor.update(control)

        speeds.append(speed)
        outputs.append(control)
        P_terms.append(P)
        I_terms.append(I)
        D_terms.append(D)
        errors.append(sp - speed)

    return {
        'label': label,
        'color': color,
        'speed': np.array(speeds),
        'output': np.array(outputs),
        'P': np.array(P_terms),
        'I': np.array(I_terms),
        'D': np.array(D_terms),
        'error': np.array(errors),
    }


# ─────────────────────────────────────────
#  Simulation Setup
# ─────────────────────────────────────────

dt = 0.01
t = np.arange(0, 10, dt)

# Setpoint profile: step up, hold, step down, hold
setpoint = np.zeros(len(t))
setpoint[t >= 1]  = 50    # Step to 50 RPM at t=1s
setpoint[t >= 5]  = 80    # Step to 80 RPM at t=5s
setpoint[t >= 8]  = 30    # Step to 30 RPM at t=8s

# Three tuning configurations
configs = [
    {'Kp': 0.5, 'Ki': 0.1, 'Kd': 0.05, 'label': 'Under-tuned  (Kp=0.5, Ki=0.1, Kd=0.05)', 'color': '#F44336'},
    {'Kp': 2.0, 'Ki': 1.0, 'Kd': 0.2,  'label': 'Well-tuned   (Kp=2.0, Ki=1.0, Kd=0.2)',  'color': '#4CAF50'},
    {'Kp': 8.0, 'Ki': 3.0, 'Kd': 0.01, 'label': 'Over-tuned   (Kp=8.0, Ki=3.0, Kd=0.01)', 'color': '#FF9800'},
]

results = [run_simulation(c['Kp'], c['Ki'], c['Kd'], setpoint, t, c['label'], c['color']) for c in configs]
well = results[1]  # Use well-tuned for detailed breakdown


# ─────────────────────────────────────────
#  Plotting
# ─────────────────────────────────────────

fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor('#0F1117')
gs = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.35)

DARK_BG  = '#1A1D27'
GRID_CLR = '#2A2D3A'
TEXT_CLR = '#E0E0E0'
SP_CLR   = '#00BCD4'

def style_ax(ax, title):
    ax.set_facecolor(DARK_BG)
    ax.set_title(title, color=TEXT_CLR, fontsize=10, fontweight='bold', pad=8)
    ax.tick_params(colors=TEXT_CLR, labelsize=8)
    ax.xaxis.label.set_color(TEXT_CLR)
    ax.yaxis.label.set_color(TEXT_CLR)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_CLR)
    ax.grid(True, color=GRID_CLR, linewidth=0.5, alpha=0.8)

# ── Plot 1: PID Tuning Comparison ──────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(t, setpoint, '--', color=SP_CLR, linewidth=1.5, label='Setpoint', zorder=5)
for r in results:
    ax1.plot(t, r['speed'], color=r['color'], linewidth=1.4, label=r['label'], alpha=0.9)
style_ax(ax1, 'Motor Speed Control — PID Tuning Comparison')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Motor Speed (RPM)')
ax1.legend(fontsize=8, facecolor='#1A1D27', edgecolor=GRID_CLR, labelcolor=TEXT_CLR)

# ── Plot 2: Error over Time (well-tuned) ───────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(t, well['error'], color='#E040FB', linewidth=1.2)
ax2.axhline(0, color=GRID_CLR, linewidth=0.8)
style_ax(ax2, 'Tracking Error — Well-Tuned PID')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Error (RPM)')

# ── Plot 3: Control Output ─────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(t, well['output'], color='#40C4FF', linewidth=1.2)
ax3.axhline(0, color=GRID_CLR, linewidth=0.8)
style_ax(ax3, 'Control Signal Output — Well-Tuned PID')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Control Output (u)')

# ── Plot 4: PID Term Breakdown ─────────────────────────────────────
ax4 = fig.add_subplot(gs[2, 0])
ax4.plot(t, well['P'], color='#FF5252', linewidth=1.0, label='P term')
ax4.plot(t, well['I'], color='#69F0AE', linewidth=1.0, label='I term')
ax4.plot(t, well['D'], color='#FFD740', linewidth=1.0, label='D term')
ax4.axhline(0, color=GRID_CLR, linewidth=0.8)
style_ax(ax4, 'PID Term Breakdown — P / I / D Contributions')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Term Value')
ax4.legend(fontsize=8, facecolor='#1A1D27', edgecolor=GRID_CLR, labelcolor=TEXT_CLR)

# ── Plot 5: Performance Metrics Table ─────────────────────────────
ax5 = fig.add_subplot(gs[2, 1])
ax5.set_facecolor(DARK_BG)
ax5.axis('off')

def calc_metrics(r, sp):
    final = sp[-1]
    # Rise time: 10% to 90% of first step
    step_indices = np.where(np.diff(sp) > 1)[0]
    if len(step_indices) == 0:
        return {'Rise Time': 'N/A', 'Overshoot': 'N/A', 'Steady-State Error': 'N/A'}
    
    idx_start = step_indices[0]
    target = sp[idx_start + 1]
    low, high = 0.1 * target, 0.9 * target
    try:
        t_low  = next(i for i in range(idx_start, len(r['speed'])) if r['speed'][i] >= low)
        t_high = next(i for i in range(idx_start, len(r['speed'])) if r['speed'][i] >= high)
        rise_time = (t_high - t_low) * dt
    except StopIteration:
        rise_time = float('nan')

    peak = np.max(r['speed'][idx_start:idx_start + int(3/dt)])
    overshoot = max(0, (peak - target) / target * 100)
    ss_error = abs(np.mean(r['speed'][-int(1/dt):]) - sp[-int(1/dt)])

    return {
        'Rise Time (s)':       f"{rise_time:.2f}",
        'Overshoot (%):':      f"{overshoot:.1f}",
        'Steady-State Err.:':  f"{ss_error:.2f} RPM",
    }

rows = []
names = ['Under-tuned', 'Well-tuned', 'Over-tuned']
colors_row = ['#F44336', '#4CAF50', '#FF9800']
for i, r in enumerate(results):
    m = calc_metrics(r, setpoint)
    rows.append([names[i]] + list(m.values()))

col_labels = ['Config', 'Rise Time', 'Overshoot', 'SS Error']
table = ax5.table(
    cellText=rows,
    colLabels=col_labels,
    loc='center',
    cellLoc='center',
)
table.auto_set_font_size(False)
table.set_fontsize(8.5)
table.scale(1, 2.0)

for (row, col), cell in table.get_celld().items():
    cell.set_facecolor('#252836' if row % 2 == 0 else DARK_BG)
    cell.set_edgecolor(GRID_CLR)
    cell.set_text_props(color=TEXT_CLR)
    if row == 0:
        cell.set_facecolor('#1E3A5F')
    if col == 0 and row > 0:
        cell.set_text_props(color=colors_row[row - 1], fontweight='bold')

ax5.set_title('Performance Metrics Summary', color=TEXT_CLR, fontsize=10, fontweight='bold', pad=8)

fig.suptitle('PID Motor Speed Controller — DC Motor Simulation',
             color=TEXT_CLR, fontsize=14, fontweight='bold', y=0.98)

plt.savefig('/home/claude/pid_controller/pid_simulation_output.png',
            dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("Done — pid_simulation_output.png saved.")
