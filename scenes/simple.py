import jax
import fdtdx
import jax.numpy as jnp
from loguru import logger
import pytreeclass

from extensions.source import CustomHardPlanceSource

def create_simple_simulation_scene(
    key: jax.Array,
    create_video: bool,
    backward_video: bool = False,
) -> tuple[
    fdtdx.ObjectContainer,
    fdtdx.ArrayContainer,
    fdtdx.ParameterContainer,
    fdtdx.SimulationConfig,
    fdtdx.Logger,
]:

    exp_logger = fdtdx.Logger(
        experiment_name="simulate_source",
    )

    wavelength = 1.55e-6
    period = fdtdx.constants.wavelength_to_period(wavelength)

    config = fdtdx.SimulationConfig(
        time=250e-15,
        resolution=100e-9,
        dtype=jnp.float32,
        courant_factor=0.99,
    )

    period_steps = round(period / config.time_step_duration)
    all_time_steps = list(range(config.time_steps_total))
    logger.info(f"{config.time_steps_total=}")
    logger.info(f"{period_steps=}")
    logger.info(f"{config.max_travel_distance=}")

    # gradient_config = fdtdx.GradientConfig(
    #     method="reversible",
    #     # num_checkpoints=30,
    #     # recorder=fdtdx.Recorder(
    #     #     modules=[
    #     #         fdtdx.DtypeConversion(dtype=jnp.float16),
    #     #     ]
    #     # )
    # )
    # config = config.aset("gradient_config", gradient_config)

    constraints = []

    volume = fdtdx.SimulationVolume(
        partial_real_shape=(12.0e-6, 12e-6, 12e-6),
        material=fdtdx.Material(  # Background material
            permittivity=2.0,
        )
    )

    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(thickness=10, boundary_type="pml")
    _, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    constraints.extend(c_list)

    source = fdtdx.GaussianPlaneSource(
        name="source",
        partial_grid_shape=(None, None, 1),
        partial_real_shape=(10e-6, 10e-6, None),
        fixed_E_polarization_vector=(1, 0, 0),
        wave_character=fdtdx.WaveCharacter(wavelength=1.550e-6),
        temporal_profile=fdtdx.SingleFrequencyProfile(num_startup_periods=5),
        radius=4e-6,
        std=1 / 3,
        direction="-",
    )
    constraints.extend(
        [
            source.place_relative_to(
                volume,
                axes=(0, 1, 2),
                own_positions=(0, 0, 0),
                other_positions=(0, 0, 1),
                grid_margins=(0, 0, -15),
            ),
        ]
    )
    
    material_config = {
        "air": fdtdx.Material(),
        "material": fdtdx.Material(permittivity=4)
    }
    voxel_size = config.resolution
    device = fdtdx.Device(
        partial_real_shape=(6e-6, 6e-6, 6e-6),
        materials=material_config,
        param_transforms=[],
        partial_voxel_real_shape=(voxel_size, voxel_size, voxel_size),
    )
    constraints.append(
        device.place_relative_to(
            volume,
            axes=(0, 1, 2),
            own_positions=(0, 0, 0),
            other_positions=(0, 0, 0),
        )
    )
    
    device_forward_detector = fdtdx.FieldDetector(
        name="device_fields",
        plot=False,
    )
    constraints.extend(device_forward_detector.same_position_and_size(device))
    
    
    start_time = round(25 * period / config.time_step_duration)
    end_time = round(46 * period / config.time_step_duration)
    switch = fdtdx.OnOffSwitch(fixed_on_time_steps=all_time_steps[start_time:end_time])
    # out_switch = fdtdx.OnOffSwitch(fixed_on_time_steps=all_time_steps[15*period_steps:19*period_steps])
    input_pf = fdtdx.PoyntingFluxDetector(
        name="input_pf",
        partial_grid_shape=(None, None, 1),
        direction="-",
        switch=switch,
        reduce_volume=True,
    )
    constraints.append(
        input_pf.place_relative_to(
            source,
            axes=2,
            own_positions=0,
            other_positions=0,
            grid_margins=-5,
        )
    )
    input_ex = fdtdx.FieldDetector(
        name="input_Ex",
        partial_grid_shape=(None, None, 1),
        switch=switch,
        components=["Ex"],
        reduce_volume=True,
    )
    constraints.append(
        input_ex.place_relative_to(
            source,
            axes=2,
            own_positions=0,
            other_positions=0,
            grid_margins=-5,
        )
    )
    input_eh = fdtdx.FieldDetector(
        name="input_EH",
        partial_grid_shape=(None, None, 1),
        switch=switch,
        reduce_volume=False,
        plot=False,
    )
    constraints.append(
        input_eh.place_relative_to(
            source,
            axes=2,
            own_positions=0,
            other_positions=0,
            grid_margins=-5,
        )
    )
    input_energy = fdtdx.EnergyDetector(
        name="input_energy",
        partial_grid_shape=(None, None, 1),
        switch=switch,
        reduce_volume=True,
    )
    constraints.append(
        input_energy.place_relative_to(
            source,
            axes=2,
            own_positions=0,
            other_positions=0,
            grid_margins=-5,
        )
    )
    
    output_detector = fdtdx.EnergyDetector(
        name="output_energy",
        partial_grid_shape=(None, None, 1),
        partial_real_shape=(1e-6, 1e-6, None),
        switch=switch,
        reduce_volume=True,
    )
    constraints.append(
        output_detector.place_relative_to(
            volume,
            axes=(0, 1, 2),
            own_positions=(0, 0, 0),
            other_positions=(0, 0, -1),
            grid_margins=(0, 0, 15,)
        )
    )
    output_ex = fdtdx.FieldDetector(
        name="output_Ex",
        partial_grid_shape=(None, None, 1),
        partial_real_shape=(1e-6, 1e-6, None),
        switch=switch,
        components=["Ex"],
        reduce_volume=True,
    )
    constraints.append(
        output_ex.place_relative_to(
            volume,
            axes=(0, 1, 2),
            own_positions=(0, 0, 0),
            other_positions=(0, 0, -1),
            grid_margins=(0, 0, 15,)
        )
    )
    output_ey = fdtdx.FieldDetector(
        name="output_Ey",
        partial_grid_shape=(None, None, 1),
        partial_real_shape=(1e-6, 1e-6, None),
        switch=switch,
        components=["Ey"],
        reduce_volume=True,
    )
    constraints.append(
        output_ey.place_relative_to(
            volume,
            axes=(0, 1, 2),
            own_positions=(0, 0, 0),
            other_positions=(0, 0, -1),
            grid_margins=(0, 0, 15,)
        )
    )
    output_ez = fdtdx.FieldDetector(
        name="output_Ez",
        partial_grid_shape=(None, None, 1),
        partial_real_shape=(1e-6, 1e-6, None),
        switch=switch,
        components=["Ez"],
        reduce_volume=True,
    )
    constraints.append(
        output_ez.place_relative_to(
            volume,
            axes=(0, 1, 2),
            own_positions=(0, 0, 0),
            other_positions=(0, 0, -1),
            grid_margins=(0, 0, 15,)
        )
    )
    output_hx = fdtdx.FieldDetector(
        name="output_Hx",
        partial_grid_shape=(None, None, 1),
        partial_real_shape=(1e-6, 1e-6, None),
        switch=switch,
        components=["Hx"],
        reduce_volume=True,
    )
    constraints.append(
        output_hx.place_relative_to(
            volume,
            axes=(0, 1, 2),
            own_positions=(0, 0, 0),
            other_positions=(0, 0, -1),
            grid_margins=(0, 0, 15,)
        )
    )
    output_hy = fdtdx.FieldDetector(
        name="output_Hy",
        partial_grid_shape=(None, None, 1),
        partial_real_shape=(1e-6, 1e-6, None),
        switch=switch,
        components=["Hy"],
        reduce_volume=True,
    )
    constraints.append(
        output_hy.place_relative_to(
            volume,
            axes=(0, 1, 2),
            own_positions=(0, 0, 0),
            other_positions=(0, 0, -1),
            grid_margins=(0, 0, 15,)
        )
    )
    output_hz = fdtdx.FieldDetector(
        name="output_Hz",
        partial_grid_shape=(None, None, 1),
        partial_real_shape=(1e-6, 1e-6, None),
        switch=switch,
        components=["Hz"],
        reduce_volume=True,
    )
    constraints.append(
        output_hz.place_relative_to(
            volume,
            axes=(0, 1, 2),
            own_positions=(0, 0, 0),
            other_positions=(0, 0, -1),
            grid_margins=(0, 0, 15,)
        )
    )
    output_eh = fdtdx.FieldDetector(
        name="output_EH",
        partial_grid_shape=(None, None, 1),
        partial_real_shape=(1e-6, 1e-6, None),
        switch=switch,
        reduce_volume=False,
        plot=False,
    )
    constraints.append(
        output_eh.place_relative_to(
            volume,
            axes=(0, 1, 2),
            own_positions=(0, 0, 0),
            other_positions=(0, 0, -1),
            grid_margins=(0, 0, 15,)
        )
    )
    out_phasor = fdtdx.PhasorDetector(
        name="output_phasor",
        wave_characters=(fdtdx.WaveCharacter(wavelength=wavelength),),
        partial_grid_shape=(None, None, 1),
        partial_real_shape=(1e-6, 1e-6, None),
        switch=switch,
        reduce_volume=False,
        plot=False,
    )
    constraints.append(
        out_phasor.place_relative_to(
            volume,
            axes=(0, 1, 2),
            own_positions=(0, 0, 0),
            other_positions=(0, 0, -1),
            grid_margins=(0, 0, 15,)
        )
    )
    adj_source = CustomHardPlanceSource(
        name="adj_source",
        wave_character=fdtdx.WaveCharacter(wavelength=wavelength),
        partial_grid_shape=(None, None, 1),
        partial_real_shape=(1e-6, 1e-6, None),
    )
    constraints.append(
        adj_source.place_relative_to(
            volume,
            axes=(0, 1, 2),
            own_positions=(0, 0, 0),
            other_positions=(0, 0, -1),
            grid_margins=(0, 0, 15,)
        )
    )
    
    
    
    exclude_list = []
    if create_video:
        video_energy_detector = fdtdx.EnergyDetector(
            name="Energy Video",
            as_slices=True,
            switch=fdtdx.OnOffSwitch(interval=3),
            exact_interpolation=True,
            # if set to positive integer, makes plotting much faster, but can also cause instabilities
            num_video_workers=8,
        )
        constraints.extend(video_energy_detector.same_position_and_size(volume))
        exclude_list.append(video_energy_detector)
        
        if backward_video:
            backwards_video_energy_detector = fdtdx.EnergyDetector(
                name="Backwards Energy Video",
                as_slices=True,
                switch=fdtdx.OnOffSwitch(interval=3),
                inverse=True,
                exact_interpolation=True,
                # if set to positive integer, makes plotting much faster, but can also cause instabilities
                num_video_workers=8, 
            )
            constraints.extend(backwards_video_energy_detector.same_position_and_size(volume))
            exclude_list.append(backwards_video_energy_detector)

    key, subkey = jax.random.split(key)
    objects, arrays, params, config, _ = fdtdx.place_objects(
        volume=volume,
        config=config,
        constraints=constraints,
        key=subkey,
    )
    logger.info(pytreeclass.tree_summary(arrays, depth=1))
    print(pytreeclass.tree_diagram(config, depth=4))

    exp_logger.savefig(
        exp_logger.cwd,
        "setup.png",
        fdtdx.plot_setup(
            config=config,
            objects=objects,
            exclude_object_list=exclude_list,
        ),
    )
    
    return objects, arrays, params, config, exp_logger

