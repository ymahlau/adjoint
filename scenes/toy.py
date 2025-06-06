import jax
import fdtdx
import jax.numpy as jnp
from loguru import logger
import pytreeclass

from extensions.source import DipoleHardPlanceSource


def create_simple_toy_scene(
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
    # example from 
    # https://www.flexcompute.com/tidy3d/learning-center/inverse-design/Lecture-2-Inverse-Design-in-Photonics-Lecture-2-Adjoint-Method/
    exp_logger = fdtdx.Logger(
        experiment_name="2d_toy",
    )

    wavelength = 2e-6
    period = fdtdx.constants.wavelength_to_period(wavelength)

    config = fdtdx.SimulationConfig(
        time=20*period,
        resolution=40e-9,
        dtype=jnp.float32,
        courant_factor=0.99,
    )

    period_steps = round(period / config.time_step_duration)
    all_time_steps = list(range(config.time_steps_total))
    logger.info(f"{config.time_steps_total=}")
    logger.info(f"{period_steps=}")
    logger.info(f"{config.max_travel_distance=}")
    
    volume = fdtdx.SimulationVolume(
        partial_real_shape=(5.8e-6, 5.8e-6, None),
        partial_grid_shape=(None, None, 3),
    )
    
    constraints = []
    bound_cfg = fdtdx.BoundaryConfig(
        boundary_type_maxz="periodic",
        boundary_type_minz="periodic",
    )
    _, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    constraints.extend(c_list)
    
    
    source = DipoleHardPlanceSource(
        name="source",
        partial_grid_shape=(1, 1, None),
        wave_character=fdtdx.WaveCharacter(wavelength=1.550e-6),
        temporal_profile=fdtdx.SingleFrequencyProfile(num_startup_periods=5),
    )
    constraints.append(
        source.set_real_coordinates(
            axes=(0, 1),
            sides=("-", "-"),
            coordinates=(1.5e-6, 2.4e-6)
        )
    )
    source_field = fdtdx.FieldDetector(
        partial_grid_shape=(1, 1, 1),
        name="source_field",
        components=["Ez"],
    )
    constraints.append(
        source_field.set_real_coordinates(
            axes=(0, 1, 2),
            sides=("-", "-", "-"),
            coordinates=(1.5e-6, 2.4e-6, 40e-9)
        )
    )
    
    material_config = {
        "air": fdtdx.Material(),
        "material": fdtdx.Material(permittivity=2)
    }
    voxel_size = config.resolution
    device = fdtdx.Device(
        partial_real_shape=(2e-6, 2e-6, None),
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
    
    phasor_switch = fdtdx.OnOffSwitch(start_after_periods=12, period=period)
    full_phasor_Ez = fdtdx.PhasorDetector(
        name="full_phasor",
        wave_characters=(fdtdx.WaveCharacter(wavelength=wavelength),),
        components=["Ez"],
        plot=False,
        switch=phasor_switch,
    )
    constraints.extend(full_phasor_Ez.same_position_and_size(volume))
    
    measure_field = fdtdx.FieldDetector(
        partial_grid_shape=(1, 1, 1),
        name="measure_field",
        components=["Ez"],
    )
    constraints.append(measure_field.set_real_coordinates(
        axes=(0, 1, 2),
        sides=("-", "-", "-"),
        coordinates=(5e-6, 3.4e-6, 40e-9),
    ))
    measure_phasor = fdtdx.PhasorDetector(
        partial_grid_shape=(1, 1, 1),
        name="measure_phasor",
        components=["Ez"],
        plot=False,
        wave_characters=(fdtdx.WaveCharacter(wavelength=wavelength),),
        switch=phasor_switch,
    )
    constraints.append(measure_phasor.set_real_coordinates(
        axes=(0, 1, 2),
        sides=("-", "-", "-"),
        coordinates=(5e-6, 3.4e-6, 40e-9),
    ))
    
    exclude_list = []
    if create_video:
        full_field_Ez = fdtdx.FieldDetector(
            name="full_field",
            components=["Ez"],
            num_video_workers=8,
            switch=fdtdx.OnOffSwitch(interval=5),
        )
        constraints.extend(full_field_Ez.same_position_and_size(volume))
        exclude_list.append(full_field_Ez)
    
    
    
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
    
    
    
    
    


