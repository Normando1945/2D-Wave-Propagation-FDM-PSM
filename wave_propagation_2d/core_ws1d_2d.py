import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from IPython.display import display
from matplotlib.animation import FFMpegWriter
import os

##############################################################################################################################################
#################################################         FFt_src       ######################################################################
##############################################################################################################################################

class FFt_src:
    """
    This class processes a source signal using the Fast Fourier Transform (FFT) method.

    Parameters:
    src (numpy array): The source time function.
    dt (float): The time step between samples.
    nt (int): The number of time samples.
    record (str, optional): A label for the record. Default is 'Source Time Function'.

    Attributes:
    signal (numpy array): The source time function.
    dt (float): The time step between samples.
    record (str): A label for the record.
    nt (int): The number of time samples.
    """
    def __init__(self, src, dt, nt, record = 'Source Time Function'):
        self.signal = src
        self.dt = dt
        self.record = record
        self.nt = nt
    
    def fft_src(self):
        """
        Calculates the Fast Fourier Transform (FFT) of the source signal.

        Returns:
        frequencies (numpy array): The array of frequencies.
        fft (numpy array): The array of amplitudes.
        RESULT (pandas DataFrame): A DataFrame containing the frequencies and their corresponding amplitudes.
        """
        signal = self.signal
        dt = self.dt
        nt = self.nt
        
        time = np.linspace(0 * dt, nt * dt, nt)
        fft = np.fft.fft(signal)
        frequencies = np.fft.fftfreq(len(signal), dt)
        idx = np.argsort(frequencies)
        frequencies = frequencies[idx]
        fft = fft[idx]

        mask = frequencies > 0
        positive_frequencies = frequencies[mask]
        positive_amplitudes = fft[len(positive_frequencies)+1:len(fft)]

        frequencies = positive_frequencies
        fft = positive_amplitudes[0:len(frequencies)]

        max_amp_idx = np.argmax(np.abs(fft))
        corresponding_freq = frequencies[max_amp_idx]
        corresponding_amp = np.abs(fft[max_amp_idx])
        F = pd.DataFrame(frequencies, columns= ['Frequencies'])
        FFT = pd.DataFrame(np.abs(fft), columns= ['Amplitudes'])
        RESULT = pd.concat([F, FFT], axis=1, ignore_index=False)

        #_______ Plotting _______
        fig, ax = plt.subplots(2,1, figsize=(16, 9))                                                                 

        ax[0].plot(time, signal, color=(0, 0, 1), marker='o', markersize=0,                               
                    markerfacecolor='w', markeredgewidth=1, linewidth=1, alpha=0.6) # plot source time function
        ax[0].set_title(f'Source Time Function, {self.record}', fontsize=12, color=(0, 0, 1))
        ax[0].set_xlim(time[0], time[-1])
        ax[0].set_xlabel('Time (s)', fontsize=10)
        ax[0].set_ylabel('Amplitude', fontsize=10)
        ax[0].grid(which='both', axis='x', linestyle='--', alpha=0.7)    
        
        ax[1].semilogx(frequencies, np.abs(fft), color=(0, 0, 1), marker='o', markersize=0,                               
                    markerfacecolor='w', markeredgewidth=1, linewidth=1, alpha=0.6) # plot frequency and amplitude                                     
        ax[1].semilogx(corresponding_freq, corresponding_amp, color=(0, 0, 0), marker='o', markersize=5,                  
                    markerfacecolor=(0, 0, 0), markeredgewidth=1, linewidth=1, alpha=1)                                 
        ax[1].text(corresponding_freq*1.05, corresponding_amp, f'{corresponding_freq:.2f} [Hz]',                           
                fontsize=10, color=(0, 0, 0), verticalalignment='bottom')
        ax[1].set_title(f'Frequency and Amplitude [FFT], {self.record}', fontsize=12, color=(0, 0, 1))                                        
        ax[1].set_xlabel('Frequency [Hz]', rotation=0, fontsize=10)                                                            
        ax[1].set_ylabel('Amplitude', rotation=90, fontsize=10)                                                                
        ax[1].set_xlim([min(frequencies), max(frequencies)])                                                                  
        ax[1].grid(which='both', axis='x', linestyle='--', alpha=0.7)                                                     
        plt.yticks([])
        plt.tight_layout()

        return frequencies, fft, RESULT


##############################################################################################################################################
################################################       Fourirer_derivate         #############################################################
##############################################################################################################################################

class Fourier_derivate_n_order:
    """
    This class calculates the n-th order derivative of a function using the Fourier method.

    Parameters:
    f (numpy array): The input function.
    dx (float): The spacing between data points.
    norder (int, optional): The order of the derivative. Default is 2.
    """
    def __init__(self, f, dx, norder = 2):
        self.f = f
        self.dx = dx
        self.norder = norder

    def fourier_derivate(self):
        """
        Calculates the n-th order derivative of the function using the Fourier method.

        Returns:
        dfn (numpy array): The n-th order derivative of the input function.
        """
        f = self.f
        dx = self.dx
        norder = self.norder
        # Length of vector f
        nx = np.size(f)
        # Initialize k vector up to Nyquist wavenumber 
        kmax = np.pi / dx
        dk = kmax / (nx / 2)
        k = np.arange(float(nx))
        k[: int(nx/2)] = k[: int(nx/2)] * dk 
        k[int(nx/2) :] = k[: int(nx/2)] - kmax
        # Fourier derivative
        ff = np.fft.fft(f)
        ff = (1j*k)**norder * ff
        dfn = np.real(np.fft.ifft(ff))                      
        return dfn, kmax, k

##############################################################################################################################################
#################################################       animation1D       ####################################################################
##############################################################################################################################################

class animation1D:
    """
    This class simulates and animates a 1D wave propagation using the finite difference method.

    Parameters:
    nx (int): The number of spatial grid points.
    c0 (float): The initial wave velocity.
    isrc (int): The index of the source location.
    dx (float): The spatial grid spacing.
    idisp (int): The interval at which to display the animation frames.
    nt (int): The total number of time steps.
    dt (float): The time step.
    src (numpy array): The source time function.
    xmax (float): The maximum x-coordinate.
    """
    def __init__(self, nx, c0, isrc, dx, idisp, nt, dt, src, xmax):
        self.nx = nx
        self.c0 = c0
        self.isrc = isrc
        self.dx = dx
        self.idisp = idisp
        self.nt = nt
        self.dt = dt
        self.src = src
        self.xmax = xmax

    def animate(self):
        """
        Simulates and animates a 1D wave propagation using the finite difference method.

        Returns:
        None
        """
        nx = self.nx
        c0 = self.c0
        isrc = self.isrc
        dx = self.dx
        idisp = self.idisp
        nt = self.nt
        dt = self.dt
        src = self.src
        xmax = self.xmax
        # Initialize empty pressure
        # -------------------------
        p    = np.zeros(nx)  # p at time n (now)
        pold = np.zeros(nx)  # p at time n-1 (past)
        pnew = np.zeros(nx)  # p at time n+1 (present)
        d2px = np.zeros(nx)  # 2nd space derivative of p

        # Initialize model (assume homogeneous model)
        # -------------------------------------------
        c = np.zeros(nx)
        c = c + c0           # initialize wave velocity in model

        # Initialize coordinate
        # ---------------------
        x = np.arange(nx)
        x = x * dx           # coordinate in x-direction

        # -------------------------
        # PLOT (VS Code .ipynb)
        # -------------------------
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))

        # Source marker
        ax.plot(x[isrc], 0, 'r*', markersize=11)

        # Line to update  (IMPORTANT: keep the comma)
        line, = ax.plot(x, p, color=(0, 0, 1), marker='o', markersize=0,                               
                            markerfacecolor='w', markeredgewidth=1, linewidth=1, alpha=0.6)

        ax.set_xlabel('x (m)')
        ax.set_ylabel('Pressure Amplitude')
        ax.grid(False)

        # Show ONCE and keep a handle (IMPORTANT)
        handle = display(fig, display_id=True)

        # 1D Wave Propagation (Finite Difference Solution)
        # ------------------------------------------------
        for it in range(nt):

            # 2nd derivative in space
            for i in range(1, nx - 1):
                d2px[i] = (p[i + 1] - 2 * p[i] + p[i - 1]) / dx**2

            # Time Extrapolation
            # ------------------
            pnew = c**2 * dt**2 * d2px + 2 * p - pold 

            # Add Source Term at isrc
            # -----------------------
            pnew[isrc] = pnew[isrc] + src[it] / (dx) * dt**2

            # Remap Time Levels
            # -----------------
            pold, p = p, pnew

            # -----------------
            # Plot update
            # -----------------
            if (it % idisp) == 0:

                # update curve
                line.set_ydata(p)

                amp = 0.0015
                ax.set_ylim(-1.1 * amp, 1.1 * amp)
                plt.grid(False)

                # follow RIGHT-going wave (ignore near source so it moves)
                skip = 0  # points to ignore after source
                i0 = min(isrc + skip, nx - 2)
                pr = np.abs(p[i0:])

                if pr.size > 0 and pr.max() > 0:
                    idx_peak = i0 + int(np.argmax(pr))
                else:
                    idx_peak = int(np.argmax(np.abs(p)))

                # zoom window parameters
                window = 100   # in grid points
                xshift = 0    # in meters (as in your code)

                x_center = x[idx_peak]
                x_left  = x_center - window * dx - xshift
                x_right = x_center + window * dx - xshift

                x_left  = max(0.0, x_left)
                x_right = min(xmax, x_right)

                ax.set_xlim(x_left, x_right)
                ax.set_title(f"Time Step (nt) = {it}")

                # update SAME output (no extra figures)
                handle.update(fig)

        # prevents a second static render in some notebook backends
        plt.close(fig) 

##############################################################################################################################################
##############################################       Safe_animation_1DW          #############################################################
##############################################################################################################################################

class Safe_animation_1DW:
    """
    This class simulates and saves a 1D wave propagation using the finite difference method.

    Parameters:
    nx (int): The number of spatial grid points.
    c0 (float): The initial wave velocity.
    isrc (int): The index of the source location.
    dx (float): The spatial grid spacing.
    idisp (int): The interval at which to display the animation frames.
    nt (int): The total number of time steps.
    dt (float): The time step.
    src (numpy array): The source time function.
    xmax (float): The maximum x-coordinate.
    """
    def __init__(self, nx, c0, isrc, dx, idisp, nt, dt, src, xmax):
        self.nx = nx
        self.c0 = c0
        self.isrc = isrc
        self.dx = dx
        self.idisp = idisp
        self.nt = nt
        self.dt = dt
        self.src = src
        self.xmax = xmax
    
    def animate_safe(self):
        """
        Simulates and saves a 1D wave propagation using the finite difference method and saves it as an MP4 file.

        Returns:
        None
        """
        nx = self.nx
        c0 = self.c0
        isrc = self.isrc
        dx = self.dx
        idisp = self.idisp
        nt = self.nt
        dt = self.dt
        src = self.src
        xmax = self.xmax
        # Initialize empty pressure
        # -------------------------
        p    = np.zeros(nx)
        pold = np.zeros(nx)
        pnew = np.zeros(nx)
        d2px = np.zeros(nx)

        # Model
        c = np.zeros(nx) + c0

        # Coordinate
        x = np.arange(nx) * dx

        # -------------------------
        # Plot setup
        # -------------------------
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))

        ax.plot(x[isrc], 0, 'r*', markersize=11)

        line, = ax.plot(
            x, p,
            color=(0, 0, 1),
            linewidth=1,
            alpha=0.6
        )

        ax.set_xlabel('x (m)')
        ax.set_ylabel('Pressure Amplitude')
        ax.grid(False)

        amp = 0.0015
        ax.set_ylim(-1.1*amp, 1.1*amp)

        # -------------------------
        # MP4 writer
        # -------------------------
        writer = FFMpegWriter(
            fps=30,
            metadata=dict(artist="MSc. Ing. Carlos Celi"),
            bitrate=1800
        )

        # -------------------------
        # Simulation + MP4
        # -------------------------
        with writer.saving(fig, "wave_propagation.mp4", dpi=120):

            for it in range(nt):

                # 2nd derivative
                for i in range(1, nx - 1):
                    d2px[i] = (p[i + 1] - 2*p[i] + p[i - 1]) / dx**2

                # Time update
                pnew = 2*p - pold + c**2 * dt**2 * d2px

                # Source
                pnew[isrc] += src[it] / dx * dt**2

                # Shift time
                pold, p = p, pnew

                # Plot update
                if (it % idisp) == 0:

                    line.set_ydata(p)

                    # Follow wave to the right
                    skip = 0
                    i0 = min(isrc + skip, nx - 2)
                    pr = np.abs(p[i0:])

                    if pr.size > 0 and pr.max() > 0:
                        idx_peak = i0 + int(np.argmax(pr))
                    else:
                        idx_peak = int(np.argmax(np.abs(p)))

                    window = 100
                    x_center = x[idx_peak]

                    ax.set_xlim(
                        max(0.0, x_center - window*dx),
                        min(xmax, x_center + window*dx)
                    )

                    ax.set_title(
                        f"1D Wave Propagation (Finite Difference Solution), Time Step (nt) = {it}",
                        fontsize=12,
                        color=(0, 0, 1)
                    )

                    writer.grab_frame()

        plt.close(fig)
        os.system("xdg-open wave_propagation.mp4")


##############################################################################################################################################
#############################################       Finite Difference Method       ###########################################################
##############################################################################################################################################
class animation2D_FDM:
    """
    This class simulates and animates a 2D wave propagation using the finite difference method.
    """

    def __init__(self,
                 nx, nz, dx, dt, nt,
                 model_type,        # Velocity Model
                 c,                 # 2D array (nz, nx) or scalar
                 isx, isz,          # source indices
                 irx, irz,          # receiver indices arrays/lists
                 src,               # source time function
                 idisp=10,          # update interval
                 nop=3,             # 3 or 5 point operators
                 fsc=3.0,           # scale factor to plot seismograph
                 cmap=plt.cm.Grays,
                 show=True,         # show animation in notebook
                 save=False,        # save video
                 video_name=None,   # output mp4 name
                 fps=10,            # video fps
                 dpi=120,           # video dpi
                 bitrate=1800):     # video bitrate

        self.nx = nx
        self.nz = nz
        self.dx = dx
        self.dt = dt
        self.nt = nt
        self.c = c
        self.isx = isx
        self.isz = isz
        self.irx = np.array(irx, dtype=int)
        self.irz = np.array(irz, dtype=int)
        self.src = src
        self.idisp = idisp
        self.nop = nop
        self.fsc = fsc
        self.model_type = model_type
        self.cmap = cmap

        self.show = show
        self.save = save
        self.video_name = video_name
        self.fps = fps
        self.dpi = dpi
        self.bitrate = bitrate

    def animate(self):
        """
        Simulates and animates a 2D wave propagation using the finite difference method.
        """
        fsc = self.fsc
        nx, nz = self.nx, self.nz
        dx, dt, nt = self.dx, self.dt, self.nt
        isx, isz = self.isx, self.isz
        irx, irz = self.irx, self.irz
        src = self.src
        idisp = self.idisp
        nop = self.nop

        show = self.show
        save = self.save
        fps = self.fps
        dpi = self.dpi
        bitrate = self.bitrate

        # -------------------------
        # Initialize fields
        # -------------------------
        p = np.zeros((nz, nx), dtype=float)
        pold = np.zeros((nz, nx), dtype=float)
        pnew = np.zeros((nz, nx), dtype=float)

        pxx = np.zeros((nz, nx), dtype=float)
        pzz = np.zeros((nz, nx), dtype=float)

        # Velocity model
        if np.isscalar(self.c):
            c = np.full((nz, nx), float(self.c))
        else:
            c = np.array(self.c, dtype=float)
            if c.shape != (nz, nx):
                raise ValueError(f"c must have shape (nz, nx) = ({nz}, {nx}), got {c.shape}")

        # Receivers for seismograms
        nrec = len(irx)
        seis = np.zeros((nrec, nt))
        ir = np.arange(nrec)

        # Courant info
        cmax = float(np.max(c))
        print("Courant Criterion eps :")
        print(cmax * dt / dx)

        # -------------------------
        # Plot
        # -------------------------
        v = float(max(abs(np.min(src)), abs(np.max(src)))) if np.size(src) else 1.0
        if v == 0:
            v = 1.0

        t = np.arange(nt) * dt

        fig, (ax0, ax1, ax2) = plt.subplots(
            1, 3,
            figsize=(30, 8),
            gridspec_kw={'width_ratios': [1.7, 1.7, 3]},
            constrained_layout=True
        )

        fig.suptitle(
            f"2D Acoustic Wave Propagation in a Heterogeneous Medium, FINITE DIFFERENCE METHOD, nop = {nop}",
            fontsize=18, fontweight='bold', color=(0, 0, 1)
        )

        # --- Velocity model ---
        im0 = ax0.imshow(c, cmap='Spectral', aspect="auto")
        ax0.set_title(f'Velocity Model, Model = {self.model_type}')
        ax0.set_xlabel('ix')
        ax0.set_ylabel('iz')
        ax0.text(
            0.01, 0.02, "by Carlos Celi",
            transform=ax0.transAxes,
            fontsize=12, fontweight='bold',
            ha='left', va='bottom', alpha=0.8, color=(0, 0, 0)
        )
        cbar0 = fig.colorbar(im0, ax=ax0, pad=0.01, fraction=0.03)
        cbar0.set_label('Velocity')
        # ax0.set_xticks(np.arange(-0.5, nx, 1), minor=True)
        # ax0.set_yticks(np.arange(-0.5, nz, 1), minor=True)
        # ax0.grid(which='minor', color=[0.5, 0.5, 0.5], linewidth=0.3, alpha=0.3)

        # --- Wavefield ---
        im = ax1.imshow(
            pnew,
            interpolation="nearest",
            animated=True,
            vmin=-v, vmax=+v,
            cmap=self.cmap,
            origin="upper",
            aspect="auto"
        )

        ax1.scatter(irx, irz, marker='^', s=60, linewidths=1.0, color=(0, 0, 1))
        for k in range(len(irx)):
            ax1.text(
                irx[k], irz[k] * 0.8, f'ST{k+1}',
                ha='center', va='bottom', fontweight='bold', color=(0, 0, 1)
            )

        ax1.scatter([isx], [isz], marker='*', s=150, color=(0, 0, 0))
        ax1.text(
            float(isx) * 1.05, float(isz), 'Source',
            ha='left', va='center', fontweight='bold', color=(0, 0, 0)
        )

        cbar = fig.colorbar(im, ax=ax1, pad=0.01, fraction=0.03)
        cbar.set_label("Pressure Amplitude")

        ax1.set_xlabel("ix [km]")
        ax1.set_ylabel("iz")
        ax1.set_title("2D Wave Propagation")

        # --- Seismograms ---
        offset = fsc * v
        offsets = np.arange(nrec) * offset

        seis_lines = []
        for k in range(nrec):
            ln, = ax2.plot(
                t[:1], np.zeros(1) + offsets[k],
                color=(0, 0, 0), linewidth=1.0, alpha=0.9
            )
            seis_lines.append(ln)

        time_line = ax2.axvline(
            0.0, linewidth=3.0, color=[0, 0, 1], linestyle='--', alpha=0.6
        )

        ax2.set_title("Seismograms")
        ax2.set_xlim(t[0], t[-1])
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Amplitude")
        ax2.set_yticks(offsets)
        ax2.set_yticklabels([f"ST{k+1}" for k in range(nrec)])
        ax2.grid(True, alpha=0.25)

        # -------------------------
        # Display handle
        # -------------------------
        handle = None
        if show:
            handle = display(fig, display_id=True)

        # -------------------------
        # Video writer
        # -------------------------
        writer = None
        if save:
            writer = FFMpegWriter(
                fps=fps,
                metadata=dict(artist="MSc. Ing. Carlos Celi"),
                bitrate=bitrate
            )

            if self.video_name is None:
                video_name = f'wave_propagation_2D_FD_{self.model_type}.mp4'
            else:
                video_name = self.video_name

        # -------------------------
        # Helper for plot update
        # -------------------------
        def update_plot(it):
            im.set_data(pnew)
            ax1.set_title(f"2D Wave Propagation | it = {it} | max(P) = {pnew.max():.3e}")

            ti = t[:it+1]
            for k, ln in enumerate(seis_lines):
                ln.set_data(ti, seis[k, :it+1] + offsets[k])

            time_line.set_xdata([t[it], t[it]])

            if save:
                writer.grab_frame()

            if show:
                handle.update(fig)

        # -------------------------
        # Simulation loop
        # -------------------------
        def run_simulation():
            nonlocal p, pold, pnew, pxx, pzz

            for it in range(nt):

                if nop == 3:
                    # second derivative with respect to x (columns)
                    for i in range(1, nx - 1):
                        pzz[:, i] = p[:, i + 1] - 2.0 * p[:, i] + p[:, i - 1]

                    # second derivative with respect to z (rows)
                    for j in range(1, nz - 1):
                        pxx[j, :] = p[j - 1, :] - 2.0 * p[j, :] + p[j + 1, :]

                elif nop == 5:
                    for i in range(2, nx - 2):
                        pzz[:, i] = (-1.0 / 12.0) * p[:, i + 2] + (4.0 / 3.0) * p[:, i + 1] \
                                    - (5.0 / 2.0) * p[:, i] + (4.0 / 3.0) * p[:, i - 1] \
                                    - (1.0 / 12.0) * p[:, i - 2]

                    for j in range(2, nz - 2):
                        pxx[j, :] = (-1.0 / 12.0) * p[j + 2, :] + (4.0 / 3.0) * p[j + 1, :] \
                                    - (5.0 / 2.0) * p[j, :] + (4.0 / 3.0) * p[j - 1, :] \
                                    - (1.0 / 12.0) * p[j - 2, :]

                else:
                    raise ValueError("nop must be 3 or 5")

                # scale by dx^2
                pxx = pxx / (dx**2)
                pzz = pzz / (dx**2)

                # time extrapolation
                pnew = 2.0 * p - pold + (dt**2) * (c**2) * (pxx + pzz)

                # source
                pnew[isz, isx] += src[it]

                # seismograms
                seis[ir, it] = pnew[irz[ir], irx[ir]]

                # plot update
                if (it % idisp) == 0:
                    update_plot(it)

                # remap time levels
                pold, p = p, pnew.copy()

                # reset derivatives
                pxx.fill(0.0)
                pzz.fill(0.0)

        # -------------------------
        # Run simulation
        # -------------------------
        if save:
            with writer.saving(fig, video_name, dpi=dpi):
                run_simulation()
            os.system(f'xdg-open "{video_name}"')
        else:
            run_simulation()

        plt.close(fig)
        return seis

##############################################################################################################################################
##############################################       PseudoSpectral Method         ###########################################################
##############################################################################################################################################

class animation2D_PeudoSpectral:
    """
    A class for simulating and visualizing 2D acoustic wave propagation using the Pseudo-Spectral Method.
    This class includes a real-time animation and the capability to record the simulation as an MP4 video.
    """
    def __init__(self,
                 nx, nz, dx, dt, nt,
                 model_type,        # Velocity Model
                 c,                 # 2D array (nz, nx) or scalar
                 isx, isz,          # source indices
                 irx, irz,          # receiver indices arrays/lists
                 src,               # source time function
                 norder,
                 idisp=10,          # update interval
                 fsc=3.0,           # scale factor to plot seismograph
                 cmap=plt.cm.Grays,
                 show=True,         # show animation in notebook
                 save=False,        # save video
                 video_name=None,   # output mp4 name
                 fps=10,            # video fps
                 dpi=120,           # video dpi
                 bitrate=1800):     # video bitrate

        self.nx = nx
        self.nz = nz
        self.dx = dx
        self.dt = dt
        self.nt = nt
        self.c = c
        self.isx = isx
        self.isz = isz
        self.irx = np.array(irx, dtype=int)
        self.irz = np.array(irz, dtype=int)
        self.src = src
        self.idisp = idisp
        self.fsc = fsc
        self.model_type = model_type
        self.cmap = cmap
        self.norder = norder
        self.show = show
        self.save = save
        self.video_name = video_name
        self.fps = fps
        self.dpi = dpi
        self.bitrate = bitrate
    def animate_PseudoSpectral(self):
        """
        Execute the simulation and generate a real-time animation of the acoustic wave propagation in the heterogeneous medium.
        If specified, this method can also record the simulation as an MP4 video.

        :return: seis: Seismogram matrix generated during the simulation.
        """
        fsc = self.fsc
        nx, nz = self.nx, self.nz
        dx, dt, nt = self.dx, self.dt, self.nt
        isx, isz = self.isx, self.isz
        irx, irz = self.irx, self.irz
        src = self.src
        idisp = self.idisp
        norder = self.norder

        show = self.show
        save = self.save
        fps = self.fps
        dpi = self.dpi
        bitrate = self.bitrate
        # -------------------------
        # Initialize fields
        # -------------------------
        p = np.zeros((nz, nx), dtype=float)
        pold = np.zeros((nz, nx), dtype=float)
        pnew = np.zeros((nz, nx), dtype=float)
        pxx = np.zeros((nz, nx), dtype=float)
        pzz = np.zeros((nz, nx), dtype=float)
        # Velocity model
        if np.isscalar(self.c):
            c = np.full((nz, nx), float(self.c))
        else:
            c = np.array(self.c, dtype=float)
            if c.shape != (nz, nx):
                raise ValueError(f"c must have shape (nz, nx) = ({nz}, {nx}), got {c.shape}")
        # Receivers for seismograms
        nrec = len(irx)
        seis = np.zeros((nrec, nt))
        ir = np.arange(nrec)
        # Courant info
        cmax = float(np.max(c))
        print("Courant Criterion eps :")
        print(cmax * dt / dx)
        # -------------------------
        # Plot
        # -------------------------
        v = float(max(abs(np.min(src)), abs(np.max(src)))) if np.size(src) else 1.0
        if v == 0:
            v = 1.0
        t = np.arange(nt) * dt
        # -------------------------
        # Figure
        # -------------------------
        fig, (ax0, ax1, ax2) = plt.subplots(
            1, 3,
            figsize=(30, 8),
            gridspec_kw={'width_ratios': [1.7, 1.7, 3]},
            constrained_layout=True
        )
        fig.suptitle("2D Acoustic Wave Propagation in a Heterogeneous Medium, PSEUDO-SPECTRAL METHOD",
            fontsize=18, fontweight='bold', color=(0, 0, 1)
        )
        # --- Velocity model ---
        im0 = ax0.imshow(c, cmap='Spectral', aspect="auto")
        ax0.set_title(f'Velocity Model, Model = {self.model_type}')
        ax0.set_xlabel('ix')
        ax0.set_ylabel('iz')
        ax0.text(
            0.01, 0.02, "by Carlos Celi",
            transform=ax0.transAxes,
            fontsize=12, fontweight='bold',
            ha='left', va='bottom', alpha=0.8, color=(0, 0, 0)
        )
        cbar0 = fig.colorbar(im0, ax=ax0, pad=0.01, fraction=0.03)
        cbar0.set_label('Velocity')
        # --- Wavefield ---
        im = ax1.imshow(
            pnew,
            interpolation="nearest",
            animated=True,
            vmin=-v, vmax=+v,
            cmap=self.cmap,
            origin="upper",
            aspect="auto"
        )
        ax1.scatter(irx, irz, marker='^', s=60, linewidths=1.0, color=(0, 0, 1))
        for k in range(len(irx)):
            ax1.text(
                irx[k], irz[k] * 0.8, f'ST{k+1}',
                ha='center', va='bottom', fontweight='bold', color=(0, 0, 1)
            )
        ax1.scatter([isx], [isz], marker='*', s=150, color=(0, 0, 0))
        ax1.text(float(isx) * 1.05, float(isz), 'Source',
            ha='left', va='center', fontweight='bold', color=(0, 0, 0))
        cbar = fig.colorbar(im, ax=ax1, pad=0.01, fraction=0.03)
        cbar.set_label("Pressure Amplitude")
        ax1.set_xlabel("ix")
        ax1.set_ylabel("iz")
        ax1.set_title("2D Wave Propagation")
        # --- Seismograms ---
        offset = fsc * v
        offsets = np.arange(nrec) * offset
        seis_lines = []
        for k in range(nrec):
            ln, = ax2.plot(
                t[:1], np.zeros(1) + offsets[k],
                color=(0, 0, 0), linewidth=1.0, alpha=0.9
            )
            seis_lines.append(ln)

        time_line = ax2.axvline(
            0.0, linewidth=3.0, color=[0, 0, 1], linestyle='--', alpha=0.6
        )
        ax2.set_title("Seismograms")
        ax2.set_xlim(t[0], t[-1])
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Amplitude")
        ax2.set_yticks(offsets)
        ax2.set_yticklabels([f"ST{k+1}" for k in range(nrec)])
        ax2.grid(True, alpha=0.25)
        # -------------------------
        # Display handle
        # -------------------------
        handle = None
        if show:
            handle = display(fig, display_id=True)
        # -------------------------
        # Video writer
        # -------------------------
        writer = None
        if save:
            writer = FFMpegWriter(fps=fps,metadata=dict(artist="MSc. Ing. Carlos Celi"),
                bitrate=bitrate)
            if self.video_name is None:
                video_name = f'wave_propagation_2D_PsuedoSpectral_{self.model_type}.mp4'
            else:
                video_name = self.video_name
        # -------------------------
        # Helper for plot update
        # -------------------------
        def update_plot(it):
            im.set_data(pnew)
            ax1.set_title(f"2D Wave Propagation | it = {it} | max(P) = {pnew.max():.3e}")
            ti = t[:it+1]
            for k, ln in enumerate(seis_lines):
                ln.set_data(ti, seis[k, :it+1] + offsets[k])
            time_line.set_xdata([t[it], t[it]])
            if save:
                writer.grab_frame()

            if show:
                handle.update(fig)
        # -------------------------
        # Simulation loop
        # -------------------------
        def run_simulation():
            nonlocal p, pold, pnew, pxx, pzz
            for it in range(nt):
                # ----------------------------------------
                # Fourier Pseudospectral Method
                # ----------------------------------------
                # second derivative with respect to z (columns)
                for j in range(nx):
                    fd_z = Fourier_derivate_n_order(p[:, j], dx, norder)
                    pzz[:, j], _, _ = fd_z.fourier_derivate()
                # second derivative with respect to x (rows)
                for i in range(nz):
                    fd_x = Fourier_derivate_n_order(p[i, :], dx, norder)
                    pxx[i, :], _, _ = fd_x.fourier_derivate()
                # Time extrapolation
                pnew = 2.0 * p - pold + (dt**2) * (c**2) * (pxx + pzz)
                # Source
                pnew[isz, isx] += src[it]
                # Seismograms
                seis[ir, it] = pnew[irz[ir], irx[ir]]
                # Plot update
                if (it % idisp) == 0:
                    update_plot(it)
                # Remap time levels
                pold, p = p, pnew.copy()
                # Reset derivatives
                pxx.fill(0.0)
                pzz.fill(0.0)
        # -------------------------
        # Run simulation
        # -------------------------
        if save:
            with writer.saving(fig, video_name, dpi=dpi):
                run_simulation()
            os.system(f'xdg-open "{video_name}"')
        else:
            run_simulation()
        plt.close(fig)
        return seis