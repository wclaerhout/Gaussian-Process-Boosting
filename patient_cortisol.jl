### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ a93795c0-38e3-11f1-bfa9-8182dfeb44b1
using Pkg; Pkg.activate(".")

# ╔═╡ a6aa8b59-54d8-4879-bce7-1e20f5407b1b
using StatsPlots, PlutoUI; TableOfContents()

# ╔═╡ ebaf341f-398d-4cbf-8bd5-1e5e50015d01
using DataFrames

# ╔═╡ 34b2988f-3ae5-4de8-9336-5c250a783e5d
using Distributions

# ╔═╡ e2972627-9b5f-4038-8ba9-621658c06b30
using Turing

# ╔═╡ 2dff25b6-8598-4f26-80e1-17303e7c2c78
md"""
## Generating features
"""

# ╔═╡ f76de121-b119-439e-a1c1-0324587f36a1
# Should sample with some structured variability
@model function generate_data()
    # --- Covariates ---
    wake_time   ~ truncated(Normal(7.0, 1.0), 4, 10)
    time_of_day ~ Uniform(wake_time, wake_time + 14.0)
    t = time_of_day - wake_time

    heart_rate_variability ~ LogNormal(log(35), 0.4)
    resting_heart_rate     ~ truncated(Normal(70, 10), 40, 110)
    sleep_duration         ~ truncated(Normal(7.2, 1.0), 3, 10)
    sleep_quality          ~ Beta(2, 6)
    physical_activity      ~ LogNormal(log(6000), 0.5)
    caffeine_intake        ~ Bernoulli(0.7)
    food_intake            ~ Gamma(2, 2)

    # --- Base diurnal curve ---
    A ~ TriangularDist(15, 20, 17.5)
    λ ~ Normal(0.3, 0.02)
    C ~ Normal(5, 0.1)
    base = A * exp(-λ * t) + C

    # --- Additive effects ---
    β_hrv       ~ Normal(-0.05, 0.01)
    β_rhr       ~ Normal(0.08, 0.015)
    β_sleep_dur ~ Normal(-1.2, 0.2)
    β_sleep_q   ~ Normal(-3.5, 0.5)

    hrv_effect       = β_hrv       * (heart_rate_variability - 35.0)
    rhr_effect       = β_rhr       * (resting_heart_rate - 70.0)
    sleep_dur_effect = β_sleep_dur * (sleep_duration - 7.2)
    sleep_q_effect   = β_sleep_q   * (sleep_quality - 0.25)

    additive_shift = hrv_effect + rhr_effect + sleep_dur_effect + sleep_q_effect

    # --- Multiplicative effects ---
    γ_caff     ~ LogNormal(log(1.15), 0.08)
    γ_activity ~ LogNormal(log(1.08), 0.05)
    γ_food     ~ LogNormal(log(1.06), 0.04)

    caff_factor     = caffeine_intake == 1 ? γ_caff : 1.0
    activity_factor = γ_activity ^ (log(physical_activity) - log(6000.0))
    food_factor     = γ_food     ^ (food_intake - 4.0)

    multiplicative_scale = caff_factor * activity_factor * food_factor

    # --- Combine ---
    C_nadir  = 5.0
    cortisol = multiplicative_scale * (base - C_nadir) + C_nadir + additive_shift

    # --- Observation noise ---
    σ ~ truncated(Normal(0.0, 0.3), 0.0, Inf)   # tightened from 1.0
    y ~ LogNormal(log(max(cortisol, 3.0)), σ)    # floor at 3.0, not 0.1

    return y
end

# ╔═╡ e35af1f1-0e61-4650-a75b-16bea2328d30
md"""
## Generating cortisol
"""

# ╔═╡ 8dcedd4e-cf0d-4d81-a7d1-3aeb2d1c2a66
@model function base_cortisol(t)
	A ~ TriangularDist(15,20,17.5)
	λ ~ Normal(0.3, 0.02)
	C ~ Normal(5,0.1)
	return A*exp(-λ*t) + C
end


# ╔═╡ 762327bc-c7c3-4bfc-acf0-4b43ceb1cd35
begin
	base_model = base_cortisol(2)
	chain_cort = sample(base_model, Prior(), 200)
	cortisol = generated_quantities(base_model, chain_cort)
end;

# ╔═╡ d8b8a58d-70c4-432e-9ebf-cfeec251df24
histogram(cortisol, title="Base cortisol after 2 hours awake", ylabel = "counts",
		 xlabel = "Cortisol conc (μ/dl)")

# ╔═╡ 127e63ab-426a-4919-8937-6a48f589bb81
md"""
## Sample data
"""

# ╔═╡ f8d36561-81f1-4328-b4c0-3e8ee4692724
model = generate_data()

# ╔═╡ 009dea19-186f-4ee8-8c84-c805f34e4590
feature_chain = sample(model, Prior(), 200)

# ╔═╡ 2b3e2445-763e-4b54-87a5-3bd238e1a47e
generated_cortisol = generated_quantities(model, feature_chain);

# ╔═╡ 612c3b2d-716d-4bf7-9db6-0e25072ff725
histogram(generated_cortisol, bins=20)

# ╔═╡ Cell order:
# ╠═a93795c0-38e3-11f1-bfa9-8182dfeb44b1
# ╠═a6aa8b59-54d8-4879-bce7-1e20f5407b1b
# ╠═ebaf341f-398d-4cbf-8bd5-1e5e50015d01
# ╠═34b2988f-3ae5-4de8-9336-5c250a783e5d
# ╠═e2972627-9b5f-4038-8ba9-621658c06b30
# ╟─2dff25b6-8598-4f26-80e1-17303e7c2c78
# ╠═f76de121-b119-439e-a1c1-0324587f36a1
# ╟─e35af1f1-0e61-4650-a75b-16bea2328d30
# ╠═8dcedd4e-cf0d-4d81-a7d1-3aeb2d1c2a66
# ╠═762327bc-c7c3-4bfc-acf0-4b43ceb1cd35
# ╠═d8b8a58d-70c4-432e-9ebf-cfeec251df24
# ╟─127e63ab-426a-4919-8937-6a48f589bb81
# ╠═f8d36561-81f1-4328-b4c0-3e8ee4692724
# ╠═009dea19-186f-4ee8-8c84-c805f34e4590
# ╠═2b3e2445-763e-4b54-87a5-3bd238e1a47e
# ╠═612c3b2d-716d-4bf7-9db6-0e25072ff725
