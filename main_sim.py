import Simulator as s
import Floe_sim as f
import FloePlot as fplt

floe_manager = f.Floe_Manager()
sim = s.Simulator()

duration = 100 # [T * duration seconds]
sigma_a = 0.0045 # Process noise [pixels]
sigma_m = 10 # Measurement noise [pixels]

# REMOVE THESE TWO LINES (14 and 15) IF YOU WANT TO TEST BACKTRACKING, AND UNCOMMENT LINES 38-41
# Add a floe before the simulator starts
sim.add_new_floe(250, 250, 500, 500, 5) # add a floe to the simulation
sim.add_GT_states() # save the floe's first state

for t in range(duration):

    # UNCOMMENT TO REMOVE A FLOE AFTER 70 TIMESTEPS FOR PREDICTING
    # if t == 70:
    #     sim.remove_floe(0)

    if t > 0:
        sim.move_all_floes(0, 10, sigma_a) # move all floes 10 pixels down

    # UNCOMMENT TO ADD A FLOE AFTER 30 TIMESTEPS IN THE SIMULATOR FOR BACKTRACKING
    # if t == 30:
    #     new_floe = sim.add_new_floe(250, 700, 500, 500, 5)
    #     new_floe.add_GT_state() # adds the first pos to sim's GT path

    curr_raster = sim.create_raster(sigma_m) # legger til pos i floe.noisy_path
    floe_manager.predict_floes()
    floe_manager.associate(curr_raster)
    floe_manager.correct_floes()

# ------- UNCOMMENT TO do BACKTRACKING -------

# floe_manager.grid.finalise()
# fplt.visualize(floe_manager.grid)
# floe_manager.backtrack_floe(0, 30, 30) # Backtrack floe 0 from timestep 0 to 30
# sim.get_GT_backtracked_states(0, 30) # get the true backtracked trajectory for floe 0 that was added at timestep 30

# -------

fplt.plot_NEES(sim, floe_manager)
fplt.plot_NIS(floe_manager)
fplt.plot_paths_sim(sim, floe_manager)
fplt.plot_map_with_uncertainties_sim(sim, floe_manager)
fplt.plot_stats_of_floe_sim(sim, floe_manager, 0)
fplt.plot_P_evolution(floe_manager, 0)