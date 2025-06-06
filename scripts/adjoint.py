import fdtdx
import jax
import jax.numpy as jnp
from fdtdx.core.plotting.debug import debug_plot_2d

from extensions.source import CustomHardPlanceSource
from scenes.simple import create_simple_simulation_scene

def main():
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    objects, arrays, params, config, exp_logger = create_simple_simulation_scene(
        subkey, 
        create_video=True,
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
        
        out_phasors = arrays.detector_states["output_phasor"]["phasor"][0, 0]
        # energy_from_phasor = fdtdx.compute_energy(out_phasors[:3], out_phasors[3:], 1, 1, axis=0).sum() / 2
        # objective = out_energy.mean()
        
        def loss_fn(eh):
            return jnp.square(jnp.abs(eh[0])).sum()
            # return -fdtdx.compute_energy(eh[:3], eh[3:], 1, 1, axis=0).sum() / 2
        
        loss, grad_eh = jax.value_and_grad(loss_fn)(out_phasors)
        
        # save forward fields
        device_forward_fields = arrays.detector_states["device_fields"]
        
        # adjoint simulation
        objects = objects.aset("['adj_source']->amplitude_E", jnp.abs(grad_eh[:3]))
        objects = objects.aset("['adj_source']->amplitude_H", jnp.abs(grad_eh[3:]))
        objects = objects.aset("['adj_source']->phase_E", jnp.angle(grad_eh[:3]))
        objects = objects.aset("['adj_source']->phase_H", jnp.angle(grad_eh[3:]))
        objects = objects.aset("['source']->static_amplitude_factor", 0.0)
        
        _, arrays = fdtdx.run_fdtd(arrays, objects, config, key)
        
        return loss, (arrays, device_forward_fields)
    
    (loss, (arrays, forward_fields)) = jax.jit(sim_fn)(params, arrays, objects, key)
    ff = forward_fields["fields"]
    bf = arrays.detector_states["device_fields"]["fields"]
    adj_grads = (bf * ff).sum(axis=(0, 1))
    
    debug_plot_2d(adj_grads.mean(axis=0), tmp_dir=exp_logger.cwd / "figures", filename="grad_x.png")
    debug_plot_2d(adj_grads.mean(axis=1), tmp_dir=exp_logger.cwd / "figures", filename="grad_y.png")
    debug_plot_2d(adj_grads.mean(axis=2), tmp_dir=exp_logger.cwd / "figures", filename="grad_z.png")
    
    a = 1
    exp_logger.log_detectors(iter_idx=0, objects=objects, detector_states=arrays.detector_states)

if __name__ == '__main__':
    main()