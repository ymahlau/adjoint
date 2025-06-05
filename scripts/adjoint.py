import fdtdx
import jax
import jax.numpy as jnp
from fdtdx.core.plotting.debug import debug_plot_2d

from scenes.simple import create_simple_simulation_scene

def main():
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    objects, arrays, params, config, exp_logger = create_simple_simulation_scene(
        subkey, 
        create_video=False,
        backward_video=False,
    )
    params = jax.tree_util.tree_map(lambda x: jnp.ones_like(x) / 2, params)
    
    def sim_fn(
        params: fdtdx.ParameterContainer,
        arrays: fdtdx.ArrayContainer,
        objects: fdtdx.ObjectContainer,
        key: jax.Array,
    ):
        key, subkey = jax.random.split(key)
        arrays, objects, _ = fdtdx.apply_params(arrays, objects, params, subkey)
        _, arrays = fdtdx.run_fdtd(arrays, objects, config, key)
        out_EH = arrays.detector_states["output_EH"]["fields"]
        out_energy = fdtdx.compute_energy(out_EH[:, :3], out_EH[:, 3:], 1, 1, axis=1).sum(axis=(1, 2, 3))
        objective = out_energy.mean()
        return -objective, (arrays,)
    
    # (loss, (arrays,)), grads = jax.jit(jax.value_and_grad(sim_fn, has_aux=True))(params, arrays, objects, key)
    
    exp_logger.log_detectors(iter_idx=0, objects=objects, detector_states=arrays.detector_states)
    

if __name__ == '__main__':
    main()