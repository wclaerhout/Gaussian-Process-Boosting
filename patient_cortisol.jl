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
@model function generate_data(n_obs::Int)

    # ------------------------------------------------------------------ #
    #  POPULATION-LEVEL HYPERPRIORS                                        #
    #  These define the distribution FROM WHICH person-traits are drawn   #
    # ------------------------------------------------------------------ #

    # Random intercept
    σ_b  ~ truncated(Normal(0.0, 1.0), 0.0, Inf)
    b    ~ Normal(0.0, σ_b)                        # person's baseline HPA set-point

    # --- Person-level random slopes (sensitivity parameters) ----------- #
    # Each γ_i is drawn from a population distribution
    # The μ is the average effect, σ is how much people differ

    # Caffeine sensitivity: most people ~15% boost, some much more/less
    μ_caff ~ LogNormal(log(1.15), 0.05)            # population mean sensitivity
    σ_caff ~ truncated(Normal(0.0, 0.08), 0.0, Inf)
    γ_caff_i ~ LogNormal(log(μ_caff), σ_caff)      # THIS person's caffeine sensitivity

    # Exercise sensitivity: varies a lot — athletes vs sedentary differ strongly
    μ_activity ~ LogNormal(log(1.08), 0.04)
    σ_activity ~ truncated(Normal(0.0, 0.10), 0.0, Inf)  # wider: more inter-individual spread
    γ_activity_i ~ LogNormal(log(μ_activity), σ_activity)

    # Fasting/food sensitivity
    μ_food ~ LogNormal(log(1.06), 0.03)
    σ_food ~ truncated(Normal(0.0, 0.06), 0.0, Inf)
    γ_food_i ~ LogNormal(log(μ_food), σ_food)

    # --- Person-level stable traits ------------------------------------ #
    resting_heart_rate     ~ truncated(Normal(70, 10), 40, 110)
    heart_rate_variability ~ LogNormal(log(35), 0.4)
    sleep_quality          ~ Beta(2, 6)

    # Person-level additive sensitivities (random slopes for stable traits)
    β_hrv_i   ~ Normal(-0.05, 0.015)   # some people's cortisol is more HRV-coupled
    β_rhr_i   ~ Normal(0.08, 0.02)
    β_sleep_q_i ~ Normal(-3.5, 0.8)   # wider SD: sleep sensitivity varies a lot

    person_shift = β_hrv_i   * (heart_rate_variability - 35.0) +
                   β_rhr_i   * (resting_heart_rate - 70.0) +
                   β_sleep_q_i * (sleep_quality - 0.25)

    # ------------------------------------------------------------------ #
    #  OBSERVATION-LEVEL LOOP                                             #
    #  Person-level γ_i and b are fixed above; only stimuli vary here     #
    # ------------------------------------------------------------------ #
    y = Vector{Float64}(undef, n_obs)

    for j in 1:n_obs
        wake_time         ~ truncated(Normal(7.0, 1.0), 4, 10)
        time_of_day       ~ Uniform(wake_time, wake_time + 14.0)
        t                  = time_of_day - wake_time
        sleep_duration    ~ truncated(Normal(7.2, 1.0), 3, 10)
        physical_activity ~ LogNormal(log(6000), 0.5)
        caffeine_intake   ~ Bernoulli(0.7)
        food_intake       ~ Gamma(2, 2)

        # Base diurnal curve
        A ~ TriangularDist(15, 20, 17.5)
        λ ~ Normal(0.3, 0.02)
        C ~ Normal(5, 0.1)
        base = A * exp(-λ * t) + C

        β_sleep_dur_i ~ Normal(-1.2, 0.3)          # random slope: sleep debt sensitivity
        sleep_dur_effect = β_sleep_dur_i * (sleep_duration - 7.2)

        # Use THIS person's sensitivity parameters, not population averages
        caff_factor     = caffeine_intake == 1 ? γ_caff_i : 1.0
        activity_factor = γ_activity_i ^ (log(physical_activity) - log(6000.0))
        food_factor     = γ_food_i     ^ (food_intake - 4.0)

        multiplicative_scale = caff_factor * activity_factor * food_factor

        C_nadir  = 5.0
        cortisol = multiplicative_scale * (base - C_nadir) + C_nadir +
                   person_shift + sleep_dur_effect + b

        σ    ~ truncated(Normal(0.0, 0.3), 0.0, Inf)
        y[j] ~ LogNormal(log(max(cortisol, 3.0)), σ)
    end

    return y
end

# ╔═╡ e35af1f1-0e61-4650-a75b-16bea2328d30
md"""
## Generating cortisol
"""

# ╔═╡ 127e63ab-426a-4919-8937-6a48f589bb81
md"""
## Sample data
"""

# ╔═╡ f8d36561-81f1-4328-b4c0-3e8ee4692724
model = generate_data(1)

# ╔═╡ 009dea19-186f-4ee8-8c84-c805f34e4590
begin
	feature_chain1 = sample(model, Prior(), 200);
	feature_chain2 = sample(model, Prior(), 200);
end;

# ╔═╡ 2b3e2445-763e-4b54-87a5-3bd238e1a47e
begin
	cortisol_person1 = generated_quantities(model, feature_chain1)
	hi = [only(g) for g in cortisol_person1]
	histogram(hi, bins=20, title = "Histogram of person1",
		 xlabel = "Cortisol (µg/dl)", label = false)
	cortisol_person2 = generated_quantities(model, feature_chain2)
	hi2 = [only(g) for g in cortisol_person2]
	histogram!(hi2, bins=20, color=:orange, alpha = 0.7, label=false)
end

# ╔═╡ Cell order:
# ╠═a93795c0-38e3-11f1-bfa9-8182dfeb44b1
# ╠═a6aa8b59-54d8-4879-bce7-1e20f5407b1b
# ╠═ebaf341f-398d-4cbf-8bd5-1e5e50015d01
# ╠═34b2988f-3ae5-4de8-9336-5c250a783e5d
# ╠═e2972627-9b5f-4038-8ba9-621658c06b30
# ╟─2dff25b6-8598-4f26-80e1-17303e7c2c78
# ╠═f76de121-b119-439e-a1c1-0324587f36a1
# ╟─e35af1f1-0e61-4650-a75b-16bea2328d30
# ╟─127e63ab-426a-4919-8937-6a48f589bb81
# ╠═f8d36561-81f1-4328-b4c0-3e8ee4692724
# ╠═009dea19-186f-4ee8-8c84-c805f34e4590
# ╠═2b3e2445-763e-4b54-87a5-3bd238e1a47e
