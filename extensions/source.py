import fdtdx
import jax
import jax.numpy as jnp

from fdtdx.objects.sources.source import Source


@fdtdx.autoinit
class CustomHardPlanceSource(Source):
    amplitude_E: jax.Array | None = None
    phase_E: jax.Array | None = None
    amplitude_H: jax.Array | None = None
    phase_H: jax.Array | None = None

    def update_E(
        self,
        E: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
        time_step: jax.Array,
        inverse: bool,
    ) -> jax.Array:
        del inv_permittivities, inv_permeabilities, inverse
        if self.amplitude_E is None:
            return E
        delta_t = self._config.time_step_duration
        time_phase = 2 * jnp.pi * time_step * delta_t / self.wave_character.period + self.wave_character.phase_shift
        time_phase = time_phase + self.phase_E
        magnitude = jnp.real(jnp.exp(-1j * time_phase))
        new_E = magnitude * self.amplitude_E
        E = E.at[:, *self.grid_slice].set(new_E)
        return E

    def update_H(
        self,
        H: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
        time_step: jax.Array,
        inverse: bool,
    ):
        del inv_permeabilities, inv_permittivities, inverse
        if self.amplitude_H is None:
            return H
        delta_t = self._config.time_step_duration
        time_phase = 2 * jnp.pi * time_step * delta_t / self.wave_character.period + self.wave_character.phase_shift
        time_phase = time_phase + self.phase_H
        magnitude = jnp.real(jnp.exp(-1j * time_phase))
        new_H = magnitude * self.amplitude_H
        H = H.at[:, *self.grid_slice].set(new_H)
        return H
