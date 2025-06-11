import Simulator as s
import Floe_sim as f
import FloePlot as fplt

floe_manager = f.Floe_Manager()
sim = s.Simulator()

duration = 100 # [T * duration seconds]
sigma_a = 0.0045 # Process noise [pixels]
sigma_m = 10 # Measurement noise [pixels]

sim.add_new_floe(250, 250, 500, 500, 5) # add a floe to the simulation
sim.add_GT_states() # save the floe's first state

for t in range(duration):    
    # if t == 70:
    #     sim.remove_floe(0)

    if t > 0:
        sim.move_all_floes(0, 10, sigma_a) # move all floes 10 pixels down

    # if t == 30:
    #     new_floe = sim.add_new_floe(250, 700, 500, 500, 5)
    #     new_floe.add_GT_state() # adds the first pos to sim's GT path

    curr_raster = sim.create_raster(sigma_m) # legger til pos i floe.noisy_path
    floe_manager.predict_floes()
    floe_manager.associate(curr_raster)
    floe_manager.correct_floes()

# floe_manager.grid.finalise()
# fplt.visualize(floe_manager.grid)
# floe_manager.backtrack_floe(0, 30, 30)
# sim.get_GT_backtracked_states(0, 30)

fplt.plot_NEES(sim, floe_manager)
fplt.plot_NIS(floe_manager)
fplt.plot_paths_sim(sim, floe_manager)
fplt.plot_map_with_uncertainties_sim(sim, floe_manager)
fplt.plot_stats_of_floe_sim(sim, floe_manager, 0)
fplt.plot_P_evolution(floe_manager, 0)