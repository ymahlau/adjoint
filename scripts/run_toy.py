import fdtdx
import jax
import jax.numpy as jnp
from fdtdx.core.plotting.debug import debug_plot_2d

from scenes.toy import create_simple_toy_scene

def main():
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    objects, arrays, params, config, exp_logger = create_simple_toy_scene(
        subkey,
        create_video=True,
        backward_video=False,
    )
    params = jax.tree_util.tree_map(lambda x: jnp.ones_like(x), params)
    
    def sim_fn(
        params: fdtdx.ParameterContainer,
        arrays: fdtdx.ArrayContainer,
        objects: fdtdx.ObjectContainer,
        key: jax.Array,
    ):
        key, subkey = jax.random.split(key)
        arrays, objects, _ = fdtdx.apply_params(arrays, objects, params, subkey)
        _, arrays = fdtdx.run_fdtd(arrays, objects, config, key)
        
        return arrays
    
    arrays = jax.jit(sim_fn)(params, arrays, objects, key)
    
    full_phasors = arrays.detector_states["full_phasor"]["phasor"][0, 0, 0, :, :, 1]
    
    debug_plot_2d(jnp.real(full_phasors), tmp_dir=exp_logger.cwd / "figures", filename="real_ez.png", cmap="seismic", center_zero=True)
    
    exp_logger.log_detectors(iter_idx=0, objects=objects, detector_states=arrays.detector_states)
    a = 1
    
if __name__ == '__main__':
    # with jax.disable_jit():
    main()