
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import numpy as np
import numpy.typing as npt

def animate_ankle_kinematics(
        t: npt.NDArray,
        alpha: npt.NDArray,
        beta: npt.NDArray,
        gamma: npt.NDArray,
        e1: npt.NDArray,
        e2: npt.NDArray,
        e3: npt.NDArray,
        o_T: npt.NDArray,
        o_C: npt.NDArray,
        T_T: npt.NDArray,
        T_C: npt.NDArray,
        markers: npt.NDArray,
        speed: float = 1.0,
        animation_fps: int = 30,
        filename: Optional[str] = None
) -> None:
    """Creates animation of the ankle joint coordinate system with corresponding kinematics.

    Args:
        t (npt.NDArray): Time vector (n_frames,)
        alpha (npt.NDArray): Dorsiflexion/plantarflexion angle trajectory (n_frames,)
        beta (npt.NDArray): Inversion/eversion angle trajectory (n_frames,)
        gamma (npt.NDArray): Internal/external rotation angle trajectory (n_frames,)
        e1 (npt.NDArray): Dorsiflexion/plantarflexion ankle axis (n_frames,3)
        e2 (npt.NDArray): Inversion/eversion ankle axis (n_frames,3)
        e3 (npt.NDArray): Internal/external rotation ankle axis (n_frames,3)
        o_T (npt.NDArray): Tibia/fibula frame origin (n_frames,3)
        o_C (npt.NDArray): Calcaneus frame origin (n_frames,3)
        T_T (npt.NDArray): Tibia/fibula frame representation (n_frames,4,4)
        T_C (npt.NDArray): Calcaneus frame representation (n_frames,4,4)
        markers (npt.NDArray): Marker trajectories to include in animation (n_frames, n_markers, 3)
        speed (float, optional): Playback speed scale. Defaults to 1.0.
        animation_fps (int, optional): Animation frames per second. Defaults to 30.
        filename (Optional[str], optional): Filename/path to save animation. Defaults to None.
    """
    
    fs_data = 1 / np.mean(np.diff(t))
    fs_animation = animation_fps / speed
    assert fs_data >= fs_animation, f"Invalid speed & animation rate: data rate is {fs_data:.01f} Hz but {fs_animation:.01f} Hz is required."

    frame_step = fs_data / fs_animation
    ani_f_idx = np.round(np.arange(0, t.shape[0], frame_step)).astype(int)

    fig = plt.figure(figsize=(16,9), facecolor='white')
    gs = gridspec.GridSpec(3, 2, width_ratios=[1, 2])

    ax_alpha = fig.add_subplot(gs[0, 0])
    ax_beta = fig.add_subplot(gs[1, 0])
    ax_gamma = fig.add_subplot(gs[2, 0])

    ax3d = fig.add_subplot(gs[:, 1], projection='3d')
    ax3d.set_axis_off()

    l_alpha, = ax_alpha.plot([], [], 'b-', lw=1.5)
    l_beta, = ax_beta.plot([], [], 'r-', lw=1.5)
    l_gamma, = ax_gamma.plot([], [], 'g-', lw=1.5)

    cursors = []
    for ax, data, title in zip(
        [ax_alpha, ax_beta, ax_gamma],
        [alpha, beta, gamma],
        ['Dorsiflexion/Plantarflexion', 'Inversion/Eversion', 'Int./Ext. Rotation']
    ):
        
        ax.set_xlim(t[0], t[-1])
        rng = np.nanmax(data) - np.nanmin(data)
        ax.set_ylim(np.nanmin(data) - 0.1*rng, np.nanmax(data) + 0.1*rng)
        ax.set_title(title, fontsize=10, loc='left')
        cursors.append(ax.axvline(0, color='k', alpha=0.5, ls=':'))

    centroids = np.nanmean(markers, axis=1) # (F, 3)
    dists = np.abs(markers - centroids[:, np.newaxis, :]) # (F, M, 3)
    r_x, r_y, r_z = np.nanmax(dists, axis=(0, 1)) * 1.2

    ax3d.set_box_aspect((r_x, r_y, r_z))
    ax3d.view_init(elev=20, azim=130)

    quivers = []
    scatters = []
    floor = None

    def update(frame_i):
        nonlocal floor

        cx, cy, cz = centroids[frame_i] # marker centroid coords
        ax3d.set_xlim(cx - r_x, cx + r_x) # update axis limits ("follow camera" on centroid)
        ax3d.set_ylim(cy - r_y, cy + r_y)
        ax3d.set_zlim(cz - r_z, cz + r_z)

        if floor:
            floor.remove()
        grid_x = np.linspace(cx - r_x, cx + r_x, 10)
        grid_y = np.linspace(cy - r_y, cy + r_y, 10)
        X, Y = np.meshgrid(grid_x, grid_y)
        Z = np.zeros_like(X) # assumes ground is at z=0
        floor = ax3d.plot_wireframe(X, Y, Z, color='gray', alpha=0.3, linewidth=0.5)

        for s in scatters:
            s.remove()
        scatters.clear()
        s = ax3d.scatter(markers[frame_i, :, 0], markers[frame_i, :, 1], markers[frame_i, :, 2], c='b', s=15, alpha=0.6)
        scatters.append(s)

        def add_vec(o: npt.NDArray, v: npt.NDArray, color: str, lw: float = 2, length: float = 40) -> None:
            """Helper for drawing 3D vectors.

            Args:
                o (npt.NDArray): Vector origin (3,)
                v (npt.NDArray): Vector (3,)
                color (str): Plotting color.
                lw (float, optional): Vector linewidth. Defaults to 2.
                length (float, optional): Vector length. Defaults to 40.
            """

            q = ax3d.quiver(
                o[0], o[1], o[2],
                v[0], v[1], v[2],
                color=color, linewidth=lw, length=length, arrow_length_ratio=0.2
            )
            quivers.append(q)
        
        for q in quivers:
            q.remove()
        quivers.clear()
        add_vec(o_T[frame_i], T_T[frame_i, :3, 0], 'gray', lw=1, length=30) # tibia/fibula x-axis
        add_vec(o_T[frame_i], T_T[frame_i, :3, 1], 'gray', lw=1, length=30) # tibia/fibula y-axis
        add_vec(o_T[frame_i], e1[frame_i], 'blue', lw=3, length=50) # e1 (alpha: dorsiflexion)
        add_vec(o_C[frame_i], T_C[frame_i, :3, 0], 'gray', lw=1, length=30) # calcaneus x-axis
        add_vec(o_C[frame_i], T_C[frame_i, :3, 2], 'gray', lw=1, length=30) # calcaneus z-axis
        add_vec(o_C[frame_i], e3[frame_i], 'green', lw=3, length=50) # e3 (gamma: internal rotation)
        add_vec((o_T[frame_i]+o_C[frame_i])/2, e2[frame_i], 'red', lw=3, length=50) # e2 at average of bone axes (beta: inversion)

        l_alpha.set_data(t[:frame_i+1], alpha[:frame_i+1])
        l_beta.set_data(t[:frame_i+1], beta[:frame_i+1])
        l_gamma.set_data(t[:frame_i+1], gamma[:frame_i+1])

        for c in cursors:
            c.set_xdata([t[frame_i]])

        return l_alpha, l_beta, l_gamma
    
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=ani_f_idx,
        interval=1000/animation_fps,
        blit=False
    )

    if filename:
        print(f"Saving to {filename}")
        writer = 'pillow' if filename.endswith('.gif') else 'ffmpeg'
        anim.save(filename, writer=writer, fps=animation_fps)
    else:
        plt.show()