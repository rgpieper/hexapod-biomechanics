
from typing import Optional, List
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
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
        T_T: npt.NDArray,
        T_C: npt.NDArray,
        COM_F: npt.NDArray,
        markers: npt.NDArray,
        side: int,
        speed: float = 1.0,
        animation_fps: int = 30,
        filename: Optional[str] = None
) -> animation.FuncAnimation:
    """Creates animation of the ankle joint coordinate system with corresponding kinematics.

    Args:
        t (npt.NDArray): Time vector [sec] (n_frames,)
        alpha (npt.NDArray): Dorsiflexion/plantarflexion angle trajectory [rad] (n_frames,)
        beta (npt.NDArray): Inversion/eversion angle trajectory [rad] (n_frames,)
        gamma (npt.NDArray): Internal/external rotation angle trajectory [rad] (n_frames,)
        e1 (npt.NDArray): Dorsiflexion/plantarflexion ankle axis (n_frames,3)
        e2 (npt.NDArray): Inversion/eversion ankle axis (n_frames,3)
        e3 (npt.NDArray): Internal/external rotation ankle axis (n_frames,3)
        T_T (npt.NDArray): Tibia/fibula frame representation [mm] (n_frames,4,4)
        T_C (npt.NDArray): Calcaneus frame representation [mm] (n_frames,4,4)
        COM_F (npt.NDArray): Foot center of mass [mm] (n_frames,3)
        markers (npt.NDArray): Marker trajectories to include in animation [mm] (n_frames, n_markers, 3)
        side (int): Ankle side (1: right, -1: left)
        speed (float, optional): Playback speed scale. Defaults to 1.0.
        animation_fps (int, optional): Animation frames per second. Defaults to 30.
        filename (Optional[str], optional): Filename/path to save animation. Defaults to None.

    Returns:
        animation.FuncAnimation: Kinematics animation object for viewing.
    """
    
    fs_data = 1 / np.mean(np.diff(t))
    fs_animation = animation_fps / speed
    assert fs_data >= fs_animation, f"Invalid speed & animation rate: data rate is {fs_data:.01f} Hz but {fs_animation:.01f} Hz is required."

    frame_step = fs_data / fs_animation
    ani_f_idx = np.round(np.arange(0, t.shape[0], frame_step)).astype(int)

    alpha = np.degrees(alpha) # [deg]
    beta = np.degrees(beta)
    gamma = np.degrees(gamma)

    fig = plt.figure(figsize=(16,9), facecolor='white')
    fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05, wspace=0.15)
    gs = gridspec.GridSpec(3, 2, width_ratios=[1, 1])

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
    if side == 1:
        ax3d.view_init(elev=20, azim=50)
    else:
        ax3d.view_init(elev=20, azim=-50)

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
        s = ax3d.scatter(markers[frame_i, :, 0], markers[frame_i, :, 1], markers[frame_i, :, 2], c='k', s=15, alpha=0.6)
        scatters.append(s)
        s_com = ax3d.scatter(COM_F[frame_i, 0], COM_F[frame_i, 1], COM_F[frame_i, 2], c='purple', s=25, alpha=0.5)
        scatters.append(s_com)

        def add_vec(o: npt.NDArray, v: npt.NDArray, color: str, lw: float = 2, length: float = 40) -> None:
            """Helper for drawing 3D vectors.

            Args:
                o (npt.NDArray): Vector origin [mm] (3,)
                v (npt.NDArray): Vector (3,)
                color (str): Plotting color.
                lw (float, optional): Vector linewidth. Defaults to 2.
                length (float, optional): Vector length [mm]. Defaults to 40.
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
        o_T = T_T[frame_i, :3, 3]
        o_C = T_T[frame_i, :3, 3]
        add_vec(o_T, T_T[frame_i, :3, 0], 'gray', lw=1, length=60) # tibia/fibula x-axis
        add_vec(o_T, T_T[frame_i, :3, 1], 'gray', lw=1, length=60) # tibia/fibula y-axis
        add_vec(o_T, e1[frame_i], 'blue', lw=3, length=100) # e1 (alpha: dorsiflexion)
        add_vec(o_C, T_C[frame_i, :3, 0], 'gray', lw=1, length=60) # calcaneus x-axis
        add_vec(o_C, T_C[frame_i, :3, 2], 'gray', lw=1, length=60) # calcaneus z-axis
        add_vec(o_C, e3[frame_i], 'green', lw=3, length=100) # e3 (gamma: internal rotation)
        add_vec((o_T+o_C)/2, e2[frame_i], 'red', lw=3, length=100) # e2 at average of bone axes (beta: inversion)

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

    plt.close()
    
    return anim

def animate_grf(
        t: npt.NDArray,
        forces: npt.NDArray,
        COP: npt.NDArray,
        moment_free: npt.NDArray,
        corners: npt.NDArray,
        sensors: npt.NDArray,
        kist_origin_base: npt.NDArray,
        side: int,
        markers: npt.NDArray,
        speed: float = 1.0,
        animation_fps: int = 30,
        stance_thresh: float = 18.0,
        force_scale: float = 0.5,
        moment_scale: float = 0.05,
        filename: Optional[str] = None
) -> animation.FuncAnimation:
    """Creates animation of the ground reaction force in context with body and hexapod position.

    Args:
        t (npt.NDArray): Time vector [sec] (n_frames,)
        forces (npt.NDArray): Component forces acting on the subject in global frame [N] (n_frames,3)
        COP (npt.NDArray): Center of pressure coordinates in global frame [mm] (n_frames,3)
        moment_free (npt.NDArray): Free moment (twist, frictional) at COP in global frame [N*mm] (n_frames,3)
        corners (npt.NDArray): Coordinate trajectories of the Kistler corners [mm] (n_frames,n_corners,3)
        sensors (npt.NDArray): Coordinate trajectories of the Kistler sensors [mm] (n_frames,n_sensors,3)
        kist_origin_base (npt.NDArray): Origin of the Kistler coordinate system in the base/static configuration [mm] (3,)
        side (int): Ankle side (1: right, -1: left)
        markers (npt.NDArray): Marker trajectories to incude in animation [mm] (n_frames,n_markers,3)
        speed (float, optional): Playback speed scale. Defaults to 1.0.
        animation_fps (int, optional): Animation frames per second. Defaults to 30.
        stance_thresh (float, optional): Vertical (z) force at which subject is in stance on Hexapod, when COP/GRF will be plotted [N]. Defaults to 18.0.
        force_scale (float, optional): Length of force vector relative to magnitude [mm/N]. Defaults to 0.5.
        moment_scale (float, optional): Length of free moment vector relative to magnitude [mm/N]. Defaults to 0.05.
        filename (Optional[str], optional): Filename/path to save animation. Defaults to None.

    Returns:
        animation.FuncAnimation: Kinematics animation object for viewing.
    """
    
    fs_data = 1 / np.mean(np.diff(t))
    fs_animation = animation_fps / speed
    assert fs_data >= fs_animation, f"Invalid speed & animation rate: data rate is {fs_data:.01f} Hz but {fs_animation:.01f} Hz is required."

    frame_step = fs_data / fs_animation
    ani_f_idx = np.round(np.arange(0, t.shape[0], frame_step)).astype(int)

    is_stance = forces[:,2] >= stance_thresh # 18N threshold

    fig = plt.figure(figsize=(16,9), facecolor='white')
    fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05, wspace=0.15)
    gs = gridspec.GridSpec(3, 2, width_ratios=[1, 1])

    ax_fx = fig.add_subplot(gs[0, 0])
    ax_fy = fig.add_subplot(gs[1, 0])
    ax_fz = fig.add_subplot(gs[2, 0])

    ax3d = fig.add_subplot(gs[:, 1], projection='3d')
    ax3d.set_axis_off()

    lines = []
    cursors = []
    for ax, data, title in zip(
        [ax_fx, ax_fy, ax_fz],
        [forces[:, 0], forces[:, 1], forces[:, 2]],
        ['Fx (Med-Lat)', 'Fy (Ant-Post)', 'Fz (Vertical)']
    ):
        
        lines.append(ax.plot([], [], c='b', lw=1.5)[0])
        ax.set_xlim(t[0], t[-1])
        rng = np.nanmax(data) - np.nanmin(data)
        ax.set_ylim(np.nanmin(data) - 0.1*rng, np.nanmax(data) + 0.1*rng)
        ax.set_title(title, fontsize=10, loc='left')
        cursors.append(ax.axvline(0, color='k', alpha=0.5, ls=':'))

    z_max = np.nanmax(markers[:, :, 2]) * 1.2
    z_min = -60
    r_x = 300
    r_y = 400

    ax3d.set_xlim(kist_origin_base[0] - r_x, kist_origin_base[0] + r_x)
    ax3d.set_ylim(kist_origin_base[1] - r_y, kist_origin_base[1] + r_y)
    ax3d.set_zlim(z_min, z_max)

    ax3d.set_box_aspect((r_x, r_y, (z_max-z_min)/2))
    if side == 1:
        ax3d.view_init(elev=10, azim=0)
    else:
        ax3d.view_init(elev=10, azim=180)

    quivers = []
    scatters = []
    floor = None
    plate = None

    def update(frame_i):
        nonlocal floor
        nonlocal plate

        if floor:
            floor.remove()
        grid_x = np.linspace(kist_origin_base[0] - r_x, kist_origin_base[0] + r_x, 10)
        grid_y = np.linspace(kist_origin_base[1] - r_y, kist_origin_base[1] + r_y, 10)
        X, Y = np.meshgrid(grid_x, grid_y)
        Z = np.zeros_like(X) # assumes ground is at z=0
        floor = ax3d.plot_wireframe(X, Y, Z, color='gray', alpha=0.3, linewidth=0.5)

        if plate:
            plate.remove()
        corner_verts = [list(zip(corners[frame_i, :, 0], corners[frame_i, :, 1], corners[frame_i, :, 2]))]
        plate = Poly3DCollection(corner_verts, alpha=0.3, facecolor='gray', edgecolor='k')
        ax3d.add_collection3d(plate)

        for s in scatters:
            s.remove()
        scatters.clear()
        for q in quivers:
            q.remove()
        quivers.clear()

        s_m = ax3d.scatter(markers[frame_i, :, 0], markers[frame_i, :, 1], markers[frame_i, :, 2], c='k', s=15, alpha=0.6)
        scatters.append(s_m)
        s_s = ax3d.scatter(sensors[frame_i, :, 0], sensors[frame_i, :, 1], sensors[frame_i, :, 2], c='g', s=15, alpha=0.6)
        scatters.append(s_s)

        if is_stance[frame_i]:
            o_grf = COP[frame_i] # (3,)
            v_f = forces[frame_i] # (3,)
            v_m = moment_free[frame_i] # (3,)

            s_c = ax3d.scatter(o_grf[0], o_grf[1], o_grf[2], c='gold', s=40, edgecolors='k', zorder=10)
            scatters.append(s_c)

            q_f = ax3d.quiver(
                o_grf[0], o_grf[1], o_grf[2],
                v_f[0], v_f[1], v_f[2],
                color='b', lw=2,
                length=np.linalg.norm(v_f) * force_scale,
                arrow_length_ratio=0.2,
                normalize=True
            )
            quivers.append(q_f)
            q_m = ax3d.quiver(
                o_grf[0], o_grf[1], o_grf[2],
                v_m[0], v_m[1], v_m[2],
                color='purple', lw=3,
                length=np.linalg.norm(v_m) * moment_scale,
                arrow_length_ratio=0.2,
                normalize=True
            )
            quivers.append(q_m)

        for l, f_dim in zip(lines, [forces[:, 0], forces[:, 1], forces[:, 2]]):
            l.set_data(t[:frame_i+1], f_dim[:frame_i+1])

        return lines
    
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
    
    plt.close()

    return anim

def animate_perturbation(
        t: npt.NDArray,
        alpha: npt.NDArray,
        M_e1: npt.NDArray,
        o_ajc: npt.NDArray,
        e1: npt.NDArray,
        o_rot: npt.NDArray,
        v_rot: npt.NDArray,
        corners: npt.NDArray,
        sensors: npt.NDArray,
        kist_origin_base: npt.NDArray,
        side: int,
        markers: npt.NDArray,
        M_components: Optional[List[npt.NDArray]] = None,
        speed: float = 1.0,
        animation_fps: int = 30,
        filename: Optional[str] = None
) -> animation.FuncAnimation:
    """Creates animation tracking ankle dorsiflexion angle and torque during perturbations with tracking of dorsiflexion axis and perturbation axis for comparison.

    Args:
        t (npt.NDArray): Time vector [sec] (n_frames,)
        alpha (npt.NDArray): Dorsiflexion/plantarflexion angle trajectory [rad] (n_frames,)
        M_e1 (npt.NDArray): Ankle moment about ankle joint coordinate system axis e1 (dorsiflexion) [N*mm] (n_frames,3)
        o_ajc (npt.NDArray): Ankle anatomical joint center trajectory in the global frame [mm] (n_frames,3)
        e1 (npt.NDArray): Ankle joint coordinate system axis e1 (dorsiflexion) (n_frames,3)
        o_rot (npt.NDArray): Origin/reference point of perturbation axis of rotation in the global frame [mm] (n_frames,3)
        v_rot (npt.NDArray): Vector describing perturbation axis of rotation (n_frames,3)
        corners (npt.NDArray): Trajectory locations of Kistler corners in the global frame [mm] (n_frames,n_corners,3)
        sensors (npt.NDArray): Trajectory locations of Kistler sensors in the global frame [mm] (n_frames,n_sensors,3)
        kist_origin_base (npt.NDArray): Origin of the Kistler frame in the base configuration, represented in the global frame [mm] (3,)
        side (int): Ankle side (1: right, -1: left)
        markers (npt.NDArray): Marker trajectories to include in animation [mm] (n_frames,n_markers,3)
        M_components (Optional[List[npt.NDArray]], optional): Component moments contributing to total dorsiflexion ankle moment to be plotted [N*mm]. Defaults to None.
        speed (float, optional): Playback speed scale. Defaults to 1.0.
        animation_fps (int, optional): Animation frames per second. Defaults to 30.
        filename (Optional[str], optional): Filename/path to save animation. Defaults to None.

    Returns:
        animation.FuncAnimation: Perturbation animation object for viewing.
    """
    
    fs_data = 1 / np.mean(np.diff(t))
    fs_animation = animation_fps / speed
    assert fs_data >= fs_animation, f"Invalid speed & animation rate: data rate is {fs_data:.01f} Hz but {fs_animation:.01f} Hz is required."

    frame_step = fs_data / fs_animation
    ani_f_idx = np.round(np.arange(0, t.shape[0], frame_step)).astype(int)

    alpha = np.degrees(alpha)

    fig = plt.figure(figsize=(16,9), facecolor='white')
    fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05, wspace=0.15)
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1])

    ax_ang = fig.add_subplot(gs[0, 0])
    ax_mom = fig.add_subplot(gs[1, 0])

    ax3d = fig.add_subplot(gs[:, 1], projection='3d')
    ax3d.set_axis_off()

    lines = []
    cursors = []
    for ax, data, title in zip(
        [ax_ang, ax_mom],
        [alpha, M_e1],
        ['Ankle Angle', 'Ankle Moment']
    ):
        lines.append(ax.plot([], [], c='b', lw=1.5)[0])
        ax.set_xlim(t[0], t[-1])
        rng = np.nanmax(data) - np.nanmin(data)
        ax.set_ylim(np.nanmin(data) - 0.1*rng, np.nanmax(data) + 0.1*rng)
        ax.set_title(title, fontsize=10, loc='left')
        cursors.append(ax.axvline(0, color='k', alpha=0.5, ls=':'))

    line_datas = [alpha, M_e1]

    if M_components is not None:
        line_datas.extend(M_components)
        all_data = np.concatenate(line_datas, axis=0)
        rng = np.nanmax(all_data) - np.nanmax(all_data)
        ax.set_ylim(np.nanmin(all_data) - 0.1*rng, np.nanmax(all_data) + 0.1*rng)
        labels = ['GRF', 'grav', 'inertia']
        colors = ['c', 'm', 'y']
        for i, comp in enumerate(M_components):
            lines.append(ax_mom.plot([], [], c=colors[i], lw=1, label=labels[i])[0])
        ax_mom.legend(loc='upper right', fontsize='small')

    z_max = np.nanmax(markers[:, :, 2]) * 1.2
    z_min = -60
    r_x = 300
    r_y = 400

    ax3d.set_xlim(kist_origin_base[0] - r_x, kist_origin_base[0] + r_x)
    ax3d.set_ylim(kist_origin_base[1] - r_y, kist_origin_base[1] + r_y)
    ax3d.set_zlim(z_min, z_max)

    ax3d.set_box_aspect((r_x, r_y, (z_max-z_min)/2))
    if side == 1:
        ax3d.view_init(elev=10, azim=200)
    else:
        ax3d.view_init(elev=10, azim=-20)

    quivers = []
    scatters = []
    floor = None
    plate = None

    def update(frame_i):
        nonlocal floor
        nonlocal plate

        if floor:
            floor.remove()
        grid_x = np.linspace(kist_origin_base[0] - r_x, kist_origin_base[0] + r_x, 10)
        grid_y = np.linspace(kist_origin_base[1] - r_y, kist_origin_base[1] + r_y, 10)
        X, Y = np.meshgrid(grid_x, grid_y)
        Z = np.zeros_like(X) # assumes ground is at z=0
        floor = ax3d.plot_wireframe(X, Y, Z, color='gray', alpha=0.3, linewidth=0.5)

        if plate:
            plate.remove()
        corner_verts = [list(zip(corners[frame_i, :, 0], corners[frame_i, :, 1], corners[frame_i, :, 2]))]
        plate = Poly3DCollection(corner_verts, alpha=0.3, facecolor='gray', edgecolor='k')
        ax3d.add_collection3d(plate)

        for s in scatters:
            s.remove()
        scatters.clear()
        for q in quivers:
            q.remove()
        quivers.clear()

        s_m = ax3d.scatter(markers[frame_i, :, 0], markers[frame_i, :, 1], markers[frame_i, :, 2], c='k', s=15, alpha=0.6)
        scatters.append(s_m)
        s_s = ax3d.scatter(sensors[frame_i, :, 0], sensors[frame_i, :, 1], sensors[frame_i, :, 2], c='g', s=15, alpha=0.6)
        scatters.append(s_s)

        o_ax_ank = o_ajc[frame_i] # (3,)
        v_ax_ank = e1[frame_i] # (3,)
        q_ank = ax3d.quiver(
                o_ax_ank[0], o_ax_ank[1], o_ax_ank[2],
                v_ax_ank[0], v_ax_ank[1], v_ax_ank[2],
                color='b', lw=3,
                length=100,
                arrow_length_ratio=0.2
            )
        quivers.append(q_ank)

        q_rot = ax3d.quiver(
                o_rot[0], o_rot[1], o_rot[2],
                v_rot[0], v_rot[1], v_rot[2],
                color='r', lw=3,
                length=200,
                arrow_length_ratio=0.2
            )
        quivers.append(q_rot)

        for l, data in zip(lines, line_datas):
            l.set_data(t[:frame_i+1], data[:frame_i+1])

        return data
    
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
    
    plt.close()

    return anim