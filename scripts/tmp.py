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
    
    (loss, (arrays,)), grads = jax.jit(jax.value_and_grad(sim_fn, has_aux=True))(params, arrays, objects, key)
    grad_arr = list(grads.values())[0]
    jnp.save(exp_logger.cwd / "grads.npy", grad_arr)
    out_EH = arrays.detector_states["output_EH"]["fields"]
    jnp.save(exp_logger.cwd / "out_EH.npy", out_EH)
    exp_logger.write({
        "loss": loss,
    })
    gt_grad_arr = jnp.load("/home/mahlau/nobackup/adjoint/outputs/saved/gt_checkpointed/grads.npy")
    grad_diff = jnp.abs(grad_arr - gt_grad_arr)
    
    debug_plot_2d(grad_arr.mean(axis=0), tmp_dir=exp_logger.cwd / "figures", filename="grad_x.png")
    debug_plot_2d(grad_arr.mean(axis=1), tmp_dir=exp_logger.cwd / "figures", filename="grad_y.png")
    debug_plot_2d(grad_arr.mean(axis=2), tmp_dir=exp_logger.cwd / "figures", filename="grad_z.png")
    
    debug_plot_2d(grad_diff.mean(axis=0), tmp_dir=exp_logger.cwd / "figures", filename="diff_x.png")
    debug_plot_2d(grad_diff.mean(axis=1), tmp_dir=exp_logger.cwd / "figures", filename="diff_y.png")
    debug_plot_2d(grad_diff.mean(axis=2), tmp_dir=exp_logger.cwd / "figures", filename="diff_z.png")
    
    exp_logger.log_detectors(iter_idx=0, objects=objects, detector_states=arrays.detector_states)
    
    # out_EH = arrays.detector_states["output_EH"]["fields"]
    # out_energy = fdtdx.compute_energy(out_EH[:, :3], out_EH[:, 3:], 1, 1, axis=1)
    # out_flux = fdtdx.compute_poynting_flux(out_EH[:, :3], out_EH[:, 3:], axis=1)
    # in_EH = arrays.detector_states["input_EH"]["fields"]
    # in_energy = fdtdx.compute_energy(in_EH[:, :3], in_EH[:, 3:], 1, 1, axis=1)
    # in_flux = fdtdx.compute_poynting_flux(in_EH[:, :3], in_EH[:, 3:], axis=1)
    

if __name__ == '__main__':
    main()