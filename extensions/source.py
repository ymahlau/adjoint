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



@fdtdx.autoinit
class DipoleHardPlanceSource(Source):
    smoothing_radius: int = fdtdx.frozen_field(default=11)

    def update_E(
        self,
        E: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
        time_step: jax.Array,
        inverse: bool,
    ) -> jax.Array:
        del inv_permittivities, inv_permeabilities, inverse
        delta_t = self._config.time_step_duration
        amplitude = self.temporal_profile.get_amplitude(
            time=delta_t * time_step,
            period=self.wave_character.period,
            phase_shift=self.wave_character.phase_shift,
        )
        center = (self.smoothing_radius - 1) / 2
        assert (self.smoothing_radius - 1) % 2 == 0
        xy = jnp.stack(jnp.meshgrid(
            jnp.arange(self.smoothing_radius) - center,
            jnp.arange(self.smoothing_radius) - center,
            indexing="xy",
        ))
        d = jnp.square(xy).sum(axis=0)
        std = center * center
        exp = jnp.exp(-d / std)
        gaussian = exp / exp.sum()
        update = jnp.zeros(shape=(3, self.smoothing_radius, self.smoothing_radius, self.grid_shape[-1]))
        update = update.at[2].set(gaussian[:, :, None])
        new_E = amplitude * update
        new_slice = (
            slice(round(self.grid_slice[0].start - center), round(self.grid_slice[0].stop + center)), 
            slice(round(self.grid_slice[1].start - center), round(self.grid_slice[1].stop + center)), 
            self.grid_slice[-1]
        )
        E = E.at[:, *new_slice].add(new_E)
        return E

    def update_H(
        self,
        H: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
        time_step: jax.Array,
        inverse: bool,
    ):
        del inv_permeabilities, inv_permittivities, inverse, time_step
        return H
        

